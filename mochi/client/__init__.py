"""Inspect an LLM client's calls in-process, with no gateway (revision item 3).

The adviser's question was whether MOCHI could be appended to an LLM's
send-request function rather than sat in front of it as a server. It can, and
this is that surface.

    from openai import OpenAI
    from mochi.client import guard

    client = guard(OpenAI())        # same client, inspected calls

Nothing else in the application changes. The wrapper inherits whatever the
application already configured - base URL, API key, timeouts, organisation -
so there is no second copy of the key, no provider address to duplicate, and
nothing to deploy.

Why both this and the proxy
---------------------------
They make different trades and the thesis should claim both honestly.

**The wrapper cannot be enforced.** It intercepts one object. An application
that builds a second client, or posts with ``requests``, never passes through
it. Anyone who can edit the code can remove one line.

**The proxy cannot be avoided**, when it is the only route out of the network,
and it works for any language because the boundary is HTTP rather than Python.

So the wrapper is the better developer experience and the weaker control. The
right recommendation depends on who is being defended against: a developer
integrating a defence wants the wrapper, an operator imposing one wants the
proxy.

What is shared
--------------
Both surfaces call the same functions - ``inspect``, ``decide``, ``apply``,
``scan_completion`` - so a verdict does not depend on which one is used. That
is a property worth testing rather than asserting, and
``tests/test_client_guard.py`` checks the two agree on identical input.
"""

from mochi.client.guard import (
    GuardedCall,
    MochiBlocked,
    guard,
    inspect_messages,
    unguard,
)

__all__ = [
    "GuardedCall",
    "MochiBlocked",
    "guard",
    "inspect_messages",
    "unguard",
]
