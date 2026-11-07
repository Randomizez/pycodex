import {mountVoiceInput} from "./voice-input.js";

// Adapt into the existing mount(host), preserving unrelated customizations.
export function mount(host) {
  return mountVoiceInput(host);
}
