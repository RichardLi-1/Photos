// When does the pill suggest the photo agent (/api/agent/message) instead of search or Ask?
// Imperative library commands: make/add albums, clean up, delete/trash, restore/undo, show albums/trash. Edit freely.
const VERBS = /^(make|create|start|build|new)\b.*\balbum\b|^album\s*:|^(add|put|save|move|copy)\b.+\b(to|into|in)\b|^(clean ?up|clean|tidy|declutter|free up)\b|^(delete|trash|remove|get rid of|bin|throw (away|out)|toss|discard|erase)\b|^(restore|undelete|recover|bring (\w+ )?back|undo|un-?trash)\b|^rename\b/i;
const SHOW = /^(show|list|open|view|see)( me)?( all)?( (of )?(my|the))? (albums?|trash|recently deleted|bin)\b|^(albums|recently deleted)$/i;
const NEED = /(don'?t|do not) (really )?need\b|what can you do/i;

export const isCommand = (text: string) => {
  const t = text.trim();
  return !!t && (VERBS.test(t) || SHOW.test(t) || NEED.test(t));
};
