import {mountSessionTimeSort} from "./session-time-sort.mjs";

// Adapt into the existing mount(host), preserving unrelated customizations.
export function mount(host) {
  return mountSessionTimeSort(host);
}
