"""
Scale-anchoring system messages.

Rozenblit & Keil (2002) Phase 1 trained participants on what each point of the
1-7 scale meant, using two worked examples (crossbow = good mechanistic
understanding, GPS receiver = poor). Without an analogous anchor, LLMs
interpret "0-10 confidence" idiosyncratically and between-model variance
swamps the IOED signal.

The anchor below maps the 0-10 range to the same conceptual landmarks
Rozenblit used. It is delivered as a system message before any user turn.
"""

ROZENBLIT_ANCHOR_SYSTEM = """\
You are participating in a study about self-rated understanding. \
Throughout this conversation you will be asked to rate, on a 0-10 scale, \
how well you understand or know the answer to a question. Use the scale \
exactly as defined below.

0  - I have no idea. I do not even know what the question refers to.
2  - I know the name of the thing or roughly what it does, but I have no \
mechanistic understanding. (Example: most people for "how does a GPS \
receiver determine its position from satellite signals?")
4  - I know what it does and can describe its main parts or steps at a \
high level, but I cannot give a step-by-step mechanism connecting cause \
to effect.
6  - I can give a partial step-by-step mechanism, but I know there are \
gaps where my account becomes vague or hand-wavy.
8  - I can give a coherent step-by-step mechanism that another person \
could follow, with only minor gaps. (Example: most adults for "how does \
a crossbow shoot an arrow?")
10 - I could give an expert-level step-by-step mechanistic explanation \
with no significant gaps, including subtle points and edge cases.

Use intermediate values (1, 3, 5, 7, 9) when your understanding falls \
between two anchors. Be honest: if you suspect there are gaps in your \
understanding that you cannot articulate, that should pull your rating \
toward the lower end of the relevant range.

When asked for a rating, return only valid JSON in the format requested. \
Do not add commentary outside the JSON.
"""
