"""Normalization and de-obfuscation layer (Phase 3).

Runs before any detector sees text: Unicode NFKC, zero-width stripping,
base64/hex/ROT13 decode-and-rescan, HTML hidden-content flagging, and file
text + metadata extraction.
"""
