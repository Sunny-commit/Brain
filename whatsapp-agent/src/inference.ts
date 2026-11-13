import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

export type SegmentationResult = {
  overlayPath: string
  tumorType: string
  regionCount: number
  confidence: number
  maskArea: number
  explanation: string
  regions: string
}

export async function analyzeMri(
  imageData: string,
  filename: string,
): Promise<SegmentationResult> {
  const workdir = await fs.mkdtemp(path.join(os.tmpdir(), 'brain-mri-'))
  const inputPath = path.join(workdir, 'input.jpg')
  const outputPath = path.join(workdir, 'overlay.png')
  const metadataPath = path.join(workdir, 'result.json')
  const imageBase64 = imageData.replace(/^data:image\/[^;]+;base64,/, '')
  const projectRoot = path.resolve(process.env.PROJECT_ROOT || '..')
  await fs.writeFile(inputPath, Buffer.from(imageBase64, 'base64'))

  try {
    await execFileAsync(
      process.env.PYTHON_EXECUTABLE || 'python',
      [
        path.join(projectRoot, 'whatsapp_inference.py'),
        inputPath,
        outputPath,
        metadataPath,
        filename,
      ],
      { cwd: projectRoot, maxBuffer: 1024 * 1024 },
    )
    const metadata = JSON.parse(await fs.readFile(metadataPath, 'utf8')) as Omit<
      SegmentationResult,
      'overlayPath'
    >
    return { ...metadata, overlayPath: outputPath }
  } catch (error) {
    await fs.rm(workdir, { recursive: true, force: true })
    throw error
  }
}

export async function cleanupAnalysis(result: SegmentationResult) {
  await fs.rm(path.dirname(result.overlayPath), { recursive: true, force: true })
}
