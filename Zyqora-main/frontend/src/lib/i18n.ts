// UI strings. Tamil is a draft and must be reviewed by a fluent Tamil speaker.
// Disposal guidance itself comes from the backend (config/voice.yaml), so the dashboard,
// the CLI and the audio clips always say the same thing.

export type Lang = "en" | "ta";

const dict = {
  classify: { en: "Classify Waste", ta: "கழிவை வகைப்படுத்து" },
  classifying: { en: "Classifying on device…", ta: "சாதனத்தில் வகைப்படுத்துகிறது…" },
  predicted: { en: "Predicted category", ta: "கணிக்கப்பட்ட வகை" },
  decision: { en: "EcoGuard decision", ta: "EcoGuard முடிவு" },
  guidance: { en: "Disposal guidance", ta: "அகற்றும் வழிகாட்டல்" },
  ACCEPT: { en: "ACCEPT", ta: "ஏற்கப்பட்டது" },
  REVIEW: { en: "REVIEW REQUIRED", ta: "சரிபார்ப்பு தேவை" },
  Organic: { en: "Organic", ta: "மக்கும் கழிவு" },
  Recyclable: { en: "Recyclable", ta: "மறுசுழற்சி செய்யக்கூடியது" },
  Residual: { en: "Residual", ta: "இதர கழிவு" },
  reviewWhy: { en: "Why review?", ta: "ஏன் சரிபார்ப்பு?" },
  noBin: { en: "No bin is opened automatically. Please ask a supervisor.", ta: "எந்த தொட்டியும் தானாகத் திறக்கப்படாது. மேற்பார்வையாளரை அழைக்கவும்." },
  play: { en: "Play instruction", ta: "அறிவுறுத்தலைக் கேட்க" },
} as const;

export type I18nKey = keyof typeof dict;

export function t(key: I18nKey, lang: Lang): string {
  return dict[key][lang];
}

export function categoryLabel(cat: string | null | undefined, lang: Lang): string {
  if (!cat) return lang === "ta" ? "இல்லை" : "None";
  return cat in dict ? t(cat as I18nKey, lang) : cat;
}
