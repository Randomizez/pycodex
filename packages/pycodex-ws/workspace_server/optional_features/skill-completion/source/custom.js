import {mountSkillCompletion} from "./skill-completion.mjs";

// Adapt into the existing mount(host), preserving unrelated customizations.
export function mount(host) {
  return mountSkillCompletion(host);
}
