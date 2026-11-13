import 'dotenv/config'
import { createServer } from 'node:http'
import qrcode from 'qrcode-terminal'
import QRCode from 'qrcode'
import WhatsApp from 'whatsapp-web.js'
import { analyzeMri, cleanupAnalysis } from './inference.js'

const { Client, LocalAuth, MessageMedia } = WhatsApp as any
let latestQr: string | null = null

const port = Number(process.env.PORT || 8080)
createServer(async (request, response) => {
  if (request.url === '/qr.png' && latestQr) {
    const png = await QRCode.toBuffer(latestQr, {
      width: 1200,
      margin: 4,
      errorCorrectionLevel: 'H',
    })
    response.writeHead(200, { 'Content-Type': 'image/png', 'Cache-Control': 'no-store' })
    response.end(png)
    return
  }

  response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
  response.end(
    latestQr
      ? '<!doctype html><title>WhatsApp QR</title><p>Scan this square QR code with WhatsApp.</p><img src="/qr.png" width="600" height="600" alt="WhatsApp QR code">'
      : '<!doctype html><title>Brain MRI WhatsApp Bridge</title><p>WhatsApp is connected or the QR code is not ready.</p>',
  )
}).listen(port, '0.0.0.0', () => {
  console.log(`QR page available on port ${port}`)
})

const client = new Client({
  authStrategy: new LocalAuth({
    clientId: process.env.WHATSAPP_CLIENT_ID || 'brain-tumor-segmentation',
    dataPath: process.env.WHATSAPP_SESSION_DIR || './.wwebjs_auth',
  }),
  puppeteer: {
    headless: true,
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || '/usr/bin/chromium',
    args: [
      '--headless=new',
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
    ],
  },
})

async function buildReply(result: Awaited<ReturnType<typeof analyzeMri>>) {
  const fallback = [
    '🧠 Brain Tumor MRI Segmentation',
    `Tumor type: ${result.tumorType}`,
    `Tumor regions: ${result.regionCount}`,
    `Confidence: ${(result.confidence * 100).toFixed(1)}%`,
    `Mask area: ${result.maskArea.toFixed(2)}%`,
    result.regions ? `Detected regions: ${result.regions}` : 'Detected regions: none',
    `Assessment: ${result.explanation}`,
    '',
    'This is an AI screening aid, not a medical diagnosis.',
  ].join('\n')
  const apiKey = process.env.GROQ_API_KEY
  if (!apiKey) return fallback

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: process.env.GROQ_MODEL || 'llama-3.3-70b-versatile',
      temperature: 0.2,
      max_tokens: 240,
      messages: [
        {
          role: 'system',
          content:
            'Rewrite the supplied MRI segmentation result clearly and cautiously for WhatsApp. Do not diagnose or invent findings. Keep the metrics unchanged and include a medical-review disclaimer.',
        },
        { role: 'user', content: fallback },
      ],
    }),
  })
  if (!response.ok) {
    console.warn(`Groq response generation failed (${response.status}); using local result`)
    return fallback
  }
  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>
  }
  return data.choices?.[0]?.message?.content?.trim() || fallback
}

async function sendAnalysis(message: any) {
  const media = await message.downloadMedia()
  if (!media?.data || !String(media.mimetype).toLowerCase().startsWith('image/')) {
    await message.reply('Please send a JPG, JPEG, or PNG MRI image.')
    return
  }

  const filename = message._data?.filename || 'whatsapp-mri.jpg'
  const result = await analyzeMri(
    `data:${media.mimetype};base64,${media.data}`,
    filename,
  )
  const reply = await buildReply(result)
  try {
    await message.reply(reply)
    const overlay = MessageMedia.fromFilePath(result.overlayPath)
    await client.sendMessage(message.from, overlay, {
      caption: 'Segmented tumor mask overlay',
    })
  } finally {
    await cleanupAnalysis(result)
  }
}

client.on('qr', (code: string) => {
  latestQr = code
  qrcode.generate(code, { small: true })
  void QRCode.toFile('whatsapp-qr.png', code, {
    width: 1200,
    margin: 4,
    errorCorrectionLevel: 'H',
  }).then(() => {
    console.log('High-resolution QR saved to whatsapp-agent/whatsapp-qr.png')
  }).catch((error: unknown) => {
    console.error('Unable to save QR PNG:', error)
  })
})
client.on('ready', () => console.log('WhatsApp connected. Send an MRI image to the linked account.'))
client.on('auth_failure', (error: string) => console.error(`WhatsApp authentication failed: ${error}`))
client.on('message', (message: any) => {
  if (message.fromMe || !message.hasMedia) return
  sendAnalysis(message).catch((error: unknown) => {
    console.error('MRI analysis failed:', error)
    message.reply('I could not analyze that image. Please try a clear JPG, JPEG, or PNG MRI scan.')
  })
})

await client.initialize()
