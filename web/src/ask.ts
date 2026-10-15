// When does the pill ask the local VLM (/api/ask, 10-60 s) instead of running a plain search (~10 ms)?
// Deliberately simple: a trailing "?" or a question-like first word. Edit freely.
const STARTS = /^(how|what|when|where|which|who|why|did|do|does|is|are|was|were|can|could|show me|find)\b/i;

export const isQuestion = (text: string) => text.trim().endsWith("?") || STARTS.test(text.trim());
