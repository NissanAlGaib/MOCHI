# Attachments: images, audio, video, documents

**Off by default, and an attachment MOCHI does not read is not inspected at
all.** If your application accepts files, turn reading on:
`read_attachments=True` for the wrapper, `MOCHI_READ_IMAGES=true` for the
proxy, and install the `[all]` extra. MOCHI then reads every attachment in the
request and inspects what it read like any other text:

| Attachment | Content block | What MOCHI reads |
|---|---|---|
| Image | `image_url` (inline `data:` only) | OCR, plus a contrast-boosted pass for pale text and a rescaled pass for images that change when shrunk; PNG/EXIF text; text hidden in the lowest bit planes (LSB steganography, PNG and other lossless formats) |
| Audio | `input_audio` | Speech transcript; ID3 tags |
| Video | `video` | Frames (OCR) and soundtrack (speech) |
| PDF, text, HTML file | `file` (inline `file_data`) | Page text, white or tiny text included; PDF metadata (Title, Keywords...); for HTML, hidden elements and comments |
| Code and config files | `file` | The whole source, comments and strings included: `.py .js .ts .java .go .rs .sh .sql .yaml` and ~60 more, or any file whose bytes are text |
| Executable | `file` | Recognised and **not** read; see [LIMITATIONS.md](LIMITATIONS.md) |
| Web page from a tool | `tool` message | The visible text and every hidden channel: `display:none`, comments, `alt`/`title` attributes |

Text a reader cannot see (pale, white, tiny or bit-plane text, and metadata)
is checked twice: once with the rest of the attachment, and once on its own.
Without the second check, a short payload hidden under a page of ordinary text
was judged together with that page and often slipped through.

A remote image URL or a `file_id` is not fetched: the bytes are not in the
request, and fetching attacker-chosen URLs would be a worse hole. Such a block
is reported as unread, not clean. `forward_mode="convert"` (the proxy's
default) replaces each attachment with the text MOCHI read, so a text-only
model can serve it and the model sees exactly what was inspected.

An attachment carried in the request that MOCHI cannot read (an executable,
an archive, a corrupt file) used to be allowed silently, because an empty
reading scores clean. It is now handled by `MOCHI_UNREADABLE_ATTACHMENTS`
(`unreadable_attachments=` for the wrapper). The values are `block`, `strip`
(replace it with a marker), and `allow` (forward it, flagged). The default is
`auto`, which strips when converting and blocks when inspecting, because an
inspecting deployment would hand the model the unread bytes.
