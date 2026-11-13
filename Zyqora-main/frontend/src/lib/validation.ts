// Client-side upload checks (the backend re-validates every image).
export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const MAX_BYTES = 10 * 1024 * 1024;

export function validateImageFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return `${file.name}: only JPEG, PNG or WebP images are supported.`;
  if (file.size === 0) return `${file.name}: the file is empty.`;
  if (file.size > MAX_BYTES) return `${file.name}: larger than 10 MB.`;
  return null;
}

