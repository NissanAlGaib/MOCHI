# Known limitations

What MOCHI does not do, and why. The README lists these in short.

- **Streaming (`stream: true`) is refused by default** (501 from the proxy,
  `MochiBlocked` from the wrapper). Outbound interception needs the complete
  response body to scan for leaked instructions and exfiltration URLs.
  `MOCHI_ALLOW_BUFFERED_STREAMING` accepts it by buffering - for chat
  completions only; the Anthropic, Gemini and Responses APIs refuse streaming.
- **The evaluation was run through chat completions.** The other SDKs reach
  the same detection code and are covered by parity tests
  (`tests/test_formats.py`) and an end-to-end check with the real SDKs, not
  by a separate attack run.
- **Stored conversations are not inspected.** A Responses API request with
  `previous_response_id` carries only the new turn; the earlier ones live at
  OpenAI. Such a request can only be forwarded to OpenAI.
- **Executables are recognised, not inspected.** A Windows, Linux or macOS
  executable attached to a request is named and flagged
  (`file_executable`), but no attempt is made to find an injection inside it.
  It is then handled by `MOCHI_UNREADABLE_ATTACHMENTS`: refused when the
  original would be forwarded, replaced with a marker when converting. Not
  implemented, deliberately:
  - a language model cannot run machine code, and the bytes it is sent are
    noise to it, so an instruction compiled into a binary does not reach the
    model as an instruction;
  - the real danger is an agent that *runs* the file, and that is malware
    analysis (signatures, sandboxing, behaviour), a different problem from
    prompt injection with mature tools of its own;
  - pulling printable strings out of a binary yields thousands of fragments.
    That is the noise the classifier is least calibrated on: one OCR fragment
    alone scored 0.61 and blocked a benign question. So it would add false
    positives for little gain.

  A deployment that hands attachments to a code-execution tool should put a
  malware scanner in front of that tool. MOCHI does not replace one.
- **The tool calls a model makes are not checked.** MOCHI inspects the text
  going into the model and the text coming back. When the model answers by
  calling a tool - `transfer_funds(from_account="1002", ...)` - nothing in
  MOCHI judges whether that action is allowed. Not implemented, deliberately:
  - there is no injection text in a tool call to detect. Whether moving money
    out of account 1002 is theft depends on who is logged in and what they
    own, which only the application knows;
  - supplying that knowledge would mean a policy for every tool on every
    deployment - a callback for the wrapper, and for the proxy a rules file
    plus a header naming the user - which turns a two-line setup into a
    security configuration;
  - and the check it would make belongs in the tool anyway. An ownership
    check inside `transfer_funds` stops the transfer even when an injection
    gets past MOCHI; a check in MOCHI does not survive MOCHI being bypassed.

  So for a tool-using application, MOCHI lowers the odds that an injection
  reaches the model, and the tools must still check authorization themselves.
  MOCHI does read the tool *list*: untrusted content that tells the model to
  run one of those tools is blocked (see [STATE_AND_TODO.md](STATE_AND_TODO.md) 4.18).
- **Remote attachments are never fetched.** An image URL or an uploaded-file
  id is flagged (`attachment_remote`) and forwarded. Fetching it would let
  any caller make MOCHI request a host of their choosing, and refusing would
  break ordinary image-URL applications. Send attachments inline to have them
  inspected.
- **Cross-provider translation covers text, images, PDFs and function
  tools.** Built-in provider tools (web search, code execution) and thinking
  blocks are forwarded only to their own provider.
