import assert from "node:assert/strict";
import {skillQuery, matchingItems} from "./skill-completion.mjs";
const draft = text => ({parts: [{type: "text", text}], selection: {start: text.length, end: text.length}});
const query = text => { const value = draft(text); return skillQuery(value.parts, value.selection); };
for (const marker of ["@", "$"]) {
  assert.equal(query(marker).prefix, "");
  assert.equal(query(`use ${marker}work`).prefix, "work");
}
for (const text of ["email@example", "foo$work", "${HOME}", "$(pwd)", "@work ", "$HOME/"]) {
  assert.equal(query(text), null);
}
assert.equal(query("中文 😀 $wor").start, "中文 😀 ".length);
assert.equal(skillQuery([{type: "context", text: "large context"},
  {type: "text", text: " $wor"}], {start: 6, end: 6}).start, 2);
assert.equal(skillQuery(draft("$work-context").parts, {start: 3, end: 3}), null);
assert.equal(skillQuery(draft("$wor").parts, {start: 0, end: 4}), null);
const items = [{name: "work-context"}, {name: "workflow"}, {name: "other"}];
assert.deepEqual(matchingItems(items, "WOR"), items.slice(0, 2));
console.log("Optional skill completion prefix checks passed.");
