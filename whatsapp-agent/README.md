# WhatsApp MRI Segmentation Bridge

This optional service uses the WhatsApp Web QR flow from the referenced
`whatsapp-ai-shopping-agent` repository and reuses the local YOLO pipeline in
the parent Brain project.

## Run

1. Copy `.env.example` to `.env`.
2. Set `PROJECT_ROOT` to the Brain repository root if starting elsewhere.
3. Install dependencies: `npm install`.
4. Start the bridge: `npm run dev`.
5. Scan the QR code printed in the terminal.
6. Send a JPG, JPEG, or PNG MRI image to the linked WhatsApp account.

The bridge replies with segmentation metrics and sends the Base64-compatible
PNG overlay as a WhatsApp image. Groq is not required for the local
segmentation path; `GROQ_API_KEY` is reserved for a future natural-language
response layer.

This is an AI screening aid, not a medical diagnosis. Keep WhatsApp session
data and MRI images private, and do not commit `.env` or authentication data.
