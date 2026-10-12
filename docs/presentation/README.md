# Proposal presentation

| File | Purpose |
|---|---|
| [`IQL_RL_Proposal.pptx`](IQL_RL_Proposal.pptx) | Visual eight-slide proposal with editable diagrams, two editable charts, concise narration and source URLs |
| [`IQL_RL_Proposal.pdf`](IQL_RL_Proposal.pdf) | PDF version of the same slides |
| [`RECORDING_GUIDE.md`](RECORDING_GUIDE.md) | Timed narration and step-by-step instructions for recording and submitting |
| [`RECORDING_SCRIPT.txt`](RECORDING_SCRIPT.txt) | Plain text narration for reading during rehearsal |
| [`REQUIREMENTS_CHECKLIST.md`](REQUIREMENTS_CHECKLIST.md) | Coverage of the supplied course proposal requirements |
| [`make_submission_zip.ps1`](make_submission_zip.ps1) | Packages one slide file with your finished narrated recording |
| [`proposal_slides.md`](proposal_slides.md) | Text source of the eight-slide proposal |
| [`final_slides.md`](final_slides.md) | Final deck with measured 100k results, video links and limitations |

The proposal targets 6 minutes 30 seconds, including time to inspect the diagrams. It covers algorithm overview (1:40), environment and protocol (2:40), and motivation with measured feasibility evidence (1:50), plus a 20-second introduction. Narration is 711 words, approximately 24% shorter than the first version. Rehearse once and keep the actual recording within 5-7 minutes.

The visual revision includes a learning-flow diagram, an editable 8/9 observation breakdown, an illustrative reward calculation, a training/selection/final-test timeline, and measured curves for all three training seeds. Slide 8 separates the reused validation curves from the independent final-test mean and stationary-tail counts. Means have no confidence bands; other training seeds still fail. Numerical reward values on slide 5 are an example, not an observed episode. Metrics are recorded evidence rather than a fresh replay.

Telman Nuruzov owns the presentation and notebook narrative. Aleksandr Medvedev checks integration and source consistency. All five team members appear by name in the deck.

The supplied handout accepts a PPTX or PDF and requires a narrated screen recording in the same ZIP. Record the presentation yourself using the guide, then create the final ZIP and upload it to Moodle. A robot rollout video alone does not meet that recording requirement.

`final_slides.md` is a separate historical 100k final-project outline. It is not the new proposal submission deck and its original measurements should not be confused with the later continuation experiments documented in `docs/POLICY_CONTINUATION.md`.
