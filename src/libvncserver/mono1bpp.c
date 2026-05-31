/*
 * mono1bpp.c - Mono 1-bpp encoding for LibVNCServer
 *
 * Encoding number : 0xFFFFFD10 (-752)
 *
 * Wire format per rectangle:
 *   [rfbFramebufferUpdateRectHeader  12 bytes]
 *   [uint8_t dither_mode              1 byte ]  0 = Floyd-Steinberg
 *   [uint8_t pixel_data[ceil(w/8)*h]          ]  MSB = leftmost pixel, rows
 *                                                 zero-padded to byte boundary
 *
 * Dithering converts any 32-bpp server pixel to 8-bit luminance (BT.601
 * weights, respecting the server's redShift/greenShift/blueShift), then
 * applies Floyd-Steinberg error diffusion before packing to 1 bit.
 *
 * C90 compatible (compiled with -std=gnu90 on Android NDK).
 */

#include <stdlib.h>
#include <string.h>

#include <rfb/rfb.h>
#include <rfb/rfbproto.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

/*
 * pixelLuma - extract 8-bit luminance from a 32-bpp server pixel.
 *
 * Uses BT.601 integer approximation:
 *   Y = (77*R + 150*G + 29*B) / 256
 * This avoids floating point entirely.
 */
static int pixelLuma(rfbClientPtr cl, uint32_t pixel)
{
    rfbPixelFormat *fmt = &cl->screen->serverFormat;
    int r, g, b;

    r = (int)((pixel >> fmt->redShift)   & fmt->redMax);
    g = (int)((pixel >> fmt->greenShift) & fmt->greenMax);
    b = (int)((pixel >> fmt->blueShift)  & fmt->blueMax);

    /* Scale each channel to 0-255 */
    if (fmt->redMax   != 255) r = (r * 255) / fmt->redMax;
    if (fmt->greenMax != 255) g = (g * 255) / fmt->greenMax;
    if (fmt->blueMax  != 255) b = (b * 255) / fmt->blueMax;

    return (77 * r + 150 * g + 29 * b) >> 8;
}

/*
 * fsDitherAndPack - Floyd-Steinberg dither an (w x h) region of the
 * server framebuffer and write the packed 1-bpp output into 'out'.
 *
 * 'out' must be at least ceil(w/8)*h bytes.
 *
 * Error diffusion kernel (fractions of 1):
 *   right       7/16
 *   lower-left  3/16
 *   below       5/16
 *   lower-right 1/16
 *
 * The error buffers are allocated on the heap to avoid large stack
 * frames on narrow devices.
 */
static rfbBool fsDitherAndPack(rfbClientPtr cl,
                               int x, int y, int w, int h,
                               uint8_t *out)
{
    int *err_cur;
    int *err_next;
    int row, col;
    int stride;
    int rowBytes;

    /* Two error rows: current scanline (err_cur) and next (err_next).
     * We need w+2 entries so that the kernel can write one past either
     * end without a bounds check. */
    err_cur  = (int *)calloc(w + 2, sizeof(int));
    err_next = (int *)calloc(w + 2, sizeof(int));
    if (!err_cur || !err_next) {
        free(err_cur);
        free(err_next);
        return FALSE;
    }

    /* Pointer offset so that err_cur[0] is column 0 and err_cur[-1] is
     * the "left bleed" slot - avoids a branch per pixel. */
    err_cur  += 1;
    err_next += 1;

    stride   = cl->screen->paddedWidthInBytes;
    rowBytes = (w + 7) >> 3;  /* ceil(w/8) */

    for (row = 0; row < h; row++) {
        uint8_t *src;
        uint8_t *outRow;
        int *tmp;

        src    = (uint8_t *)cl->screen->frameBuffer + (y + row) * stride + x * 4;
        outRow = out + row * rowBytes;

        memset(outRow, 0, rowBytes);
        memset(err_next - 1, 0, (w + 2) * sizeof(int));

        for (col = 0; col < w; col++) {
            uint32_t pixel;
            int luma;
            int luma_err;
            int bit;
            int error;

            memcpy(&pixel, src + col * 4, 4);
            luma = pixelLuma(cl, pixel);

            /* Add accumulated error, clamp to [0, 255] */
            luma_err = luma + err_cur[col];
            if (luma_err < 0)   luma_err = 0;
            if (luma_err > 255) luma_err = 255;

            /* Threshold at 128 */
            bit   = (luma_err >= 128) ? 1 : 0;
            error = luma_err - (bit ? 255 : 0);

            /* Pack into output byte, MSB = leftmost */
            if (bit) {
                outRow[col >> 3] |= (uint8_t)(0x80u >> (col & 7));
            }

            /* Distribute error to neighbours */
            err_cur [col + 1] += (error * 7) >> 4;  /* right       7/16 */
            err_next[col - 1] += (error * 3) >> 4;  /* lower-left  3/16 */
            err_next[col    ] += (error * 5) >> 4;  /* below       5/16 */
            err_next[col + 1] += (error * 1) >> 4;  /* lower-right 1/16 */
        }

        /* Swap error rows for next scanline */
        tmp      = err_cur;
        err_cur  = err_next;
        err_next = tmp;
    }

    free(err_cur  - 1);
    free(err_next - 1);
    return TRUE;
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

rfbBool rfbSendRectEncodingMono1bpp(rfbClientPtr cl,
                                    int x, int y, int w, int h)
{
    rfbFramebufferUpdateRectHeader hdr;
    uint8_t  dither_mode_byte;
    int      rowBytes;
    int      dataLen;
    uint8_t *packed;

    rowBytes = (w + 7) >> 3;
    dataLen  = rowBytes * h;

    packed = (uint8_t *)malloc(dataLen);
    if (!packed) {
        rfbLogPerror("rfbSendRectEncodingMono1bpp: malloc");
        return FALSE;
    }

    /* Dither the rectangle into the packed buffer */
    if (!fsDitherAndPack(cl, x, y, w, h, packed)) {
        free(packed);
        rfbLogPerror("rfbSendRectEncodingMono1bpp: dither OOM");
        return FALSE;
    }

    /*
     * Flush updateBuf FIRST.
     *
     * rfbSendFramebufferUpdate writes the FramebufferUpdate message header
     * (type + padding + nRects) into cl->updateBuf before calling us.
     * All other encoders that use rfbWriteExact (e.g. Raw) explicitly flush
     * updateBuf before issuing any rfbWriteExact calls.  If we skip this
     * step, our rect header and pixel data reach the client BEFORE the
     * FramebufferUpdate header that wraps them, corrupting the stream and
     * causing the client to report "server to client message type" errors.
     */
    if (cl->ublen > 0) {
        if (!rfbSendUpdateBuf(cl)) {
            free(packed);
            return FALSE;
        }
    }

    /* --- Send rectangle header --- */
    hdr.r.x      = Swap16IfLE((uint16_t)x);
    hdr.r.y      = Swap16IfLE((uint16_t)y);
    hdr.r.w      = Swap16IfLE((uint16_t)w);
    hdr.r.h      = Swap16IfLE((uint16_t)h);
    hdr.encoding = Swap32IfLE((uint32_t)rfbEncodingMono1bpp);

    if (rfbWriteExact(cl, (char *)&hdr, sz_rfbFramebufferUpdateRectHeader) < 0) {
        free(packed);
        return FALSE;
    }

    /* --- Send 1-byte dither-mode flag --- */
    dither_mode_byte = (uint8_t)rfbMono1bppDitherFloydSteinberg;
    if (rfbWriteExact(cl, (char *)&dither_mode_byte, 1) < 0) {
        free(packed);
        return FALSE;
    }

    /* --- Send packed pixel data --- */
    if (rfbWriteExact(cl, (char *)packed, dataLen) < 0) {
        free(packed);
        return FALSE;
    }

    rfbStatRecordEncodingSent(cl, rfbEncodingMono1bpp,
                              sz_rfbFramebufferUpdateRectHeader + 1 + dataLen,
                              sz_rfbFramebufferUpdateRectHeader + 1 + dataLen);

    free(packed);
    return TRUE;
}
