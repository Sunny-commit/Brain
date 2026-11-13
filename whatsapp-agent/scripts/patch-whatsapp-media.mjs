import fs from 'node:fs/promises'
import path from 'node:path'

const target = path.resolve(
  'node_modules/whatsapp-web.js/src/util/Injected/Utils.js',
)

try {
  let source = await fs.readFile(target, 'utf8')
  if (source.includes('delete message.__x_id')) {
    console.log('WhatsApp media compatibility patch already applied')
    process.exit(0)
  }

  const sendMessageStart = source.indexOf('window.WWebJS.sendMessage')
  const messageStart = source.indexOf('const message = {', sendMessageStart)
  const messageEnd = messageStart >= 0 ? source.indexOf('};', messageStart) : -1
  if (sendMessageStart < 0 || messageStart < 0 || messageEnd < 0) {
    console.warn('WhatsApp media patch skipped: library shape was not recognized')
    process.exit(0)
  }

  const insertion =
    '\n\n // WhatsApp Web media regression: preserve the outgoing message id.\n delete message.__x_id;'
  source = `${source.slice(0, messageEnd + 2)}${insertion}${source.slice(messageEnd + 2)}`
  await fs.writeFile(target, source)
  console.log('Patched WhatsApp media compatibility regression')
} catch (error) {
  console.warn(`WhatsApp media patch skipped: ${error.message}`)
}
