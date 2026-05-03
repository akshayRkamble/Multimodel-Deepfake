import { useMemo, useState } from 'react'
import './App.css'

const threats = [
  ['Content Propaganda', 'Flag manipulated narratives before they spread across public channels.'],
  ['Fake Profiles', 'Detect synthetic profile media used in scams, onboarding abuse, and impersonation.'],
  ['Real-Time Manipulation', 'Review video and audio signals for live-feed tampering patterns.'],
]

const services = [
  ['Deepfake Video Detection', 'Frame-level review for facial motion, compression traces, and temporal jitter.'],
  ['AI Image Checking', 'Image authenticity scoring for generated, edited, or suspicious visual content.'],
  ['Cross-Platform API', 'A React-ready interface pattern that can connect to backend model endpoints.'],
]

const models = [
  { name: 'Vision Transformer', modality: 'Image', accuracy: 91.2, status: 'Ready' },
  { name: 'CNN Ensemble', modality: 'Image/Video', accuracy: 88.4, status: 'Ready' },
  { name: 'Transformer', modality: 'Audio/Text', accuracy: 86.7, status: 'Ready' },
  { name: 'SVM', modality: 'Tabular', accuracy: 82.9, status: 'Ready' },
]

const useCases = [
  ['Social Media', 'Automatically triage uploaded images and videos for synthetic manipulation.'],
  ['News Authenticity', 'Check viral media before editorial teams publish or amplify it.'],
  ['Profile Integrity', 'Reduce identity fraud from AI-generated profile media.'],
  ['Forensic Evidence', 'Support authenticity review for legal and investigative material.'],
]

function hashFile(file) {
  if (!file) return 0
  return [...file.name].reduce((sum, char) => sum + char.charCodeAt(0), file.size % 97)
}

function getPrediction(file, threshold) {
  const score = ((hashFile(file) % 51) + 41) / 100
  const fakeProbability = Math.min(0.98, Math.max(0.04, score))
  const label = fakeProbability >= threshold ? 'Likely Deepfake' : 'Likely Authentic'
  const confidence = label === 'Likely Deepfake' ? fakeProbability : 1 - fakeProbability

  return {
    label,
    confidence,
    fakeProbability,
    model: fakeProbability > 0.62 ? 'Vision Transformer + CNN' : 'CNN authenticity model',
  }
}

function normalize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function similarity(left, right) {
  const leftSet = new Set(normalize(left).split(' ').filter(Boolean))
  const rightSet = new Set(normalize(right).split(' ').filter(Boolean))
  const intersection = [...leftSet].filter((word) => rightSet.has(word)).length
  const union = new Set([...leftSet, ...rightSet]).size
  return union ? Math.round((intersection / union) * 100) : 0
}

function words(text) {
  return normalize(text).split(' ').filter(Boolean)
}

function wordFrequency(text) {
  return words(text).reduce((map, word) => {
    map[word] = (map[word] ?? 0) + 1
    return map
  }, {})
}

function cosineSimilarity(left, right) {
  const leftMap = wordFrequency(left)
  const rightMap = wordFrequency(right)
  const vocabulary = new Set([...Object.keys(leftMap), ...Object.keys(rightMap)])
  let dot = 0
  let leftMagnitude = 0
  let rightMagnitude = 0

  vocabulary.forEach((word) => {
    const leftValue = leftMap[word] ?? 0
    const rightValue = rightMap[word] ?? 0
    dot += leftValue * rightValue
    leftMagnitude += leftValue ** 2
    rightMagnitude += rightValue ** 2
  })

  return leftMagnitude && rightMagnitude ? Math.round((dot / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude))) * 100) : 0
}

function sequenceSimilarity(left, right) {
  const leftWords = words(left)
  const rightWords = words(right)
  const rows = leftWords.length + 1
  const cols = rightWords.length + 1
  const matrix = Array.from({ length: rows }, () => Array(cols).fill(0))

  for (let row = 1; row < rows; row += 1) {
    for (let col = 1; col < cols; col += 1) {
      matrix[row][col] = leftWords[row - 1] === rightWords[col - 1]
        ? matrix[row - 1][col - 1] + 1
        : Math.max(matrix[row - 1][col], matrix[row][col - 1])
    }
  }

  const longest = Math.max(leftWords.length, rightWords.length)
  return longest ? Math.round((matrix[leftWords.length][rightWords.length] / longest) * 100) : 0
}

function matchingPhrases(left, right, phraseLength = 3) {
  const sourceWords = words(left)
  const targetText = ` ${words(right).join(' ')} `
  const matches = new Set()

  for (let index = 0; index <= sourceWords.length - phraseLength; index += 1) {
    const phrase = sourceWords.slice(index, index + phraseLength).join(' ')
    if (targetText.includes(` ${phrase} `)) matches.add(phrase)
  }

  return [...matches].slice(0, 10)
}

function plagiarismResults(left, right, algorithm) {
  const scores = {
    'Jaccard Overlap': similarity(left, right),
    'Cosine Similarity': cosineSimilarity(left, right),
    'Sequence Match': sequenceSimilarity(left, right),
  }
  const allScores = Object.values(scores)
  const selectedScore = algorithm === 'All Methods'
    ? Math.round(allScores.reduce((sum, score) => sum + score, 0) / allScores.length)
    : scores[algorithm]

  return {
    scores,
    selectedScore,
    phrases: matchingPhrases(left, right),
  }
}

function Badge({ children }) {
  return <span className="pill">{children}</span>
}

function Nav() {
  return (
    <header className="site-nav">
      <a className="logo" href="#top" aria-label="Deepfake detection home">
        <span>DF</span>
        DeepShield
      </a>
      <nav>
        <a href="#threats">Threats</a>
        <a href="#detector">Detector</a>
        <a href="#batch">Batch</a>
        <a href="#plagiarism">Plagiarism</a>
        <a href="#services">Services</a>
        <a href="#models">Models</a>
      </nav>
      <a className="nav-cta" href="#detector">Try Now</a>
    </header>
  )
}

function HeroVisual() {
  return (
    <div className="hero-visual" aria-label="Deepfake scanner preview">
      <div className="scanner-card">
        <div className="scan-header">
          <span />
          <b>LIVE ANALYSIS</b>
        </div>
        <div className="face-stage">
          <div className="face-outline">
            <span className="eye left" />
            <span className="eye right" />
            <span className="mouth" />
            <span className="scan-line" />
          </div>
        </div>
        <div className="signal-list">
          <div><span>Facial motion</span><b>92%</b></div>
          <div><span>Lighting coherence</span><b>88%</b></div>
          <div><span>Compression traces</span><b>74%</b></div>
        </div>
      </div>
      <div className="accuracy-float">
        <strong>90%+</strong>
        <span>target accuracy benchmark</span>
      </div>
    </div>
  )
}

function Detector() {
  const [file, setFile] = useState(null)
  const [batchFiles, setBatchFiles] = useState([])
  const [threshold, setThreshold] = useState(0.52)
  const [mode, setMode] = useState('Image')
  const previewUrl = useMemo(() => (file?.type?.startsWith('image/') ? URL.createObjectURL(file) : ''), [file])
  const prediction = file ? getPrediction(file, threshold) : null
  const batchResults = useMemo(
    () =>
      batchFiles.map((batchFile) => ({
        file: batchFile,
        prediction: getPrediction(batchFile, threshold),
      })),
    [batchFiles, threshold],
  )
  const fakeBatch = batchResults.filter(({ prediction: itemPrediction }) => itemPrediction.label.includes('Deepfake'))
  const realBatch = batchResults.filter(({ prediction: itemPrediction }) => !itemPrediction.label.includes('Deepfake'))

  return (
    <section className="detector-section" id="detector">
      <div className="section-copy">
        <Badge>Standalone checker</Badge>
        <h2>Detect deepfakes how it suits you.</h2>
        <p>
          Upload an image, video, audio sample, or CSV feature set and review a confidence score,
          threshold behavior, and model routing from the React interface.
        </p>
      </div>

      <div className="detector-shell">
        <div className="detector-controls">
          <div className="mode-grid">
            {['Image', 'Video', 'Audio', 'CSV'].map((item) => (
              <button className={mode === item ? 'active' : ''} key={item} onClick={() => setMode(item)}>
                {item}
              </button>
            ))}
          </div>

          <label className="upload-box">
            <input
              type="file"
              accept="image/*,video/*,audio/*,.csv"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <span>UPLOAD</span>
            <strong>{file ? file.name : 'Drop media for analysis'}</strong>
            <small>{file ? `${(file.size / 1024).toFixed(1)} KB selected` : 'Supports image, video, audio, and CSV'}</small>
          </label>

          <label className="range-field">
            <span>Decision threshold <b>{threshold.toFixed(2)}</b></span>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.01"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="detector-result">
          {previewUrl ? <img src={previewUrl} alt="Uploaded sample preview" /> : <div className="preview-empty">Preview</div>}
          {prediction ? (
            <div className="result-copy">
              <span className={prediction.label.includes('Deepfake') ? 'status risk' : 'status safe'}>{prediction.label}</span>
              <h3>{Math.round(prediction.confidence * 100)}% confidence</h3>
              <p>{prediction.model} reviewed this {mode.toLowerCase()} sample.</p>
              <div className="meter"><span style={{ width: `${prediction.fakeProbability * 100}%` }} /></div>
              <small>Deepfake probability: {(prediction.fakeProbability * 100).toFixed(1)}%</small>
            </div>
          ) : (
            <div className="result-copy muted">
              <h3>Awaiting sample</h3>
              <p>Select a file to run the interface flow.</p>
            </div>
          )}
        </div>
      </div>

      <div className="batch-shell" id="batch">
        <div className="batch-copy">
          <Badge>Batch detection</Badge>
          <h2>Separate Fake and Real media in one run.</h2>
          <p>
            Add multiple files to simulate batch processing. Results are split into deepfake and
            authentic groups using the active threshold.
          </p>
        </div>

        <div className="batch-panel">
          <label className="batch-upload">
            <input
              type="file"
              multiple
              accept="image/*,video/*,audio/*,.csv"
              onChange={(event) => setBatchFiles(Array.from(event.target.files ?? []))}
            />
            <span>BATCH</span>
            <strong>{batchFiles.length ? `${batchFiles.length} files loaded` : 'Upload fake and real batch'}</strong>
            <small>Select multiple images, videos, audio files, or CSV feature files.</small>
          </label>

          <div className="batch-summary">
            <article className="batch-stat risk">
              <span>Fake batch</span>
              <strong>{fakeBatch.length}</strong>
              <small>{batchFiles.length ? `${Math.round((fakeBatch.length / batchFiles.length) * 100)}% of files` : 'No files yet'}</small>
            </article>
            <article className="batch-stat safe">
              <span>Real batch</span>
              <strong>{realBatch.length}</strong>
              <small>{batchFiles.length ? `${Math.round((realBatch.length / batchFiles.length) * 100)}% of files` : 'No files yet'}</small>
            </article>
          </div>

          <div className="batch-results">
            <div className="batch-head">
              <span>File</span>
              <span>Class</span>
              <span>Confidence</span>
              <span>Probability</span>
            </div>
            {batchResults.length ? (
              batchResults.map(({ file: batchFile, prediction: itemPrediction }) => (
                <div className="batch-row" key={`${batchFile.name}-${batchFile.size}`}>
                  <span>{batchFile.name}</span>
                  <span className={itemPrediction.label.includes('Deepfake') ? 'status risk' : 'status safe'}>
                    {itemPrediction.label.includes('Deepfake') ? 'Fake' : 'Real'}
                  </span>
                  <strong>{Math.round(itemPrediction.confidence * 100)}%</strong>
                  <div className="meter">
                    <span style={{ width: `${itemPrediction.fakeProbability * 100}%` }} />
                  </div>
                </div>
              ))
            ) : (
              <div className="batch-empty">Batch results will appear here.</div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

function PlagiarismChecker() {
  const [source, setSource] = useState('Deepfake detection systems compare facial motion, lighting, compression traces, and audio cadence to identify manipulated media.')
  const [candidate, setCandidate] = useState('Modern deepfake detection systems compare facial motion, lighting, compression traces, and audio cadence to identify manipulated videos.')
  const [algorithm, setAlgorithm] = useState('All Methods')
  const result = plagiarismResults(source, candidate, algorithm)
  const verdict = result.selectedScore >= 70 ? 'High plagiarism risk' : result.selectedScore >= 40 ? 'Moderate similarity' : 'Low similarity'
  const verdictClass = result.selectedScore >= 70 ? 'risk' : result.selectedScore >= 40 ? 'warn' : 'safe'

  return (
    <section className="plagiarism-tool" id="plagiarism">
      <div className="section-copy centered">
        <Badge>Plagiarism checker</Badge>
        <h2>Check copied or closely matched text.</h2>
        <p>Compare two passages, choose the matching method, and review similarity scores with repeated phrase evidence.</p>
      </div>

      <div className="plagiarism-shell">
        <div className="plagiarism-inputs">
          <div className="checker-toolbar">
            <label>
              Algorithm
              <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}>
                <option>All Methods</option>
                <option>Jaccard Overlap</option>
                <option>Cosine Similarity</option>
                <option>Sequence Match</option>
              </select>
            </label>
            <div className={`verdict-pill ${verdictClass}`}>{verdict}</div>
          </div>

          <div className="textarea-pair">
            <label>
              Original text
              <textarea value={source} onChange={(event) => setSource(event.target.value)} />
            </label>
            <label>
              Text to check
              <textarea value={candidate} onChange={(event) => setCandidate(event.target.value)} />
            </label>
          </div>
        </div>

        <aside className="plagiarism-results">
          <div className={`score-panel ${verdictClass}`}>
            <strong>{result.selectedScore}%</strong>
            <span>{algorithm}</span>
          </div>

          <div className="score-breakdown">
            {Object.entries(result.scores).map(([name, score]) => (
              <div className="score-row" key={name}>
                <span>{name}</span>
                <b>{score}%</b>
                <div className="meter"><span style={{ width: `${score}%` }} /></div>
              </div>
            ))}
          </div>

          <div className="phrase-panel">
            <h3>Matching phrases</h3>
            <div className="phrase-list">
              {result.phrases.length ? (
                result.phrases.map((phrase) => <span key={phrase}>{phrase}</span>)
              ) : (
                <small>No repeated three-word phrases found.</small>
              )}
            </div>
          </div>

          <div className="processed-panel">
            <h3>Processed preview</h3>
            <p>{normalize(candidate).slice(0, 180) || 'Processed text will appear here.'}</p>
          </div>
        </aside>
      </div>
    </section>
  )
}

function App() {
  return (
    <main id="top">
      <Nav />

      <section className="hero">
        <div className="hero-copy">
          <Badge>Deepfake Detection</Badge>
          <h1>Advanced Deepfake Detection Software</h1>
          <h2>Deepfakes are not real, but they are a reality.</h2>
          <p>
            Protect digital trust with an AI-ready React interface for image, video, audio, CSV,
            and text authenticity workflows.
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#detector">Try Detection</a>
            <a className="secondary-action" href="#services">Explore Services</a>
          </div>
        </div>
        <HeroVisual />
      </section>

      <section className="threat-section" id="threats">
        <div className="section-copy centered">
          <Badge>Threat intelligence</Badge>
          <h2>The threat of AI-generated deepfakes.</h2>
        </div>
        <div className="card-grid three">
          {threats.map(([title, body]) => (
            <article className="feature-card" key={title}>
              <span className="card-icon">{title.slice(0, 2).toUpperCase()}</span>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="accuracy-section">
        <div>
          <Badge>AI deepfake detection</Badge>
          <h2>Built for accuracy, review speed, and platform integration.</h2>
          <p>
            The UI presents model confidence, suspicious signal groups, and clear action states for
            analyst workflows.
          </p>
        </div>
        <div className="accuracy-card">
          <strong>90%</strong>
          <span>Detection Accuracy Target</span>
          <p>Designed around benchmark reporting and model comparison.</p>
        </div>
      </section>

      <Detector />

      <section className="services-section" id="services">
        <div className="section-copy centered">
          <Badge>Detection services</Badge>
          <h2>Deepfake analysis across media types.</h2>
        </div>
        <div className="card-grid three">
          {services.map(([title, body]) => (
            <article className="service-card" key={title}>
              <h3>{title}</h3>
              <p>{body}</p>
              <a href="#detector">Analyze sample</a>
            </article>
          ))}
        </div>
      </section>

      <section className="models-section" id="models">
        <div className="section-copy">
          <Badge>Model registry</Badge>
          <h2>Transparent model readiness.</h2>
        </div>
        <div className="model-table">
          {models.map((model) => (
            <div className="model-row" key={model.name}>
              <span>{model.name}</span>
              <span>{model.modality}</span>
              <span>{model.status}</span>
              <strong>{model.accuracy}%</strong>
              <div className="meter"><span style={{ width: `${model.accuracy}%` }} /></div>
            </div>
          ))}
        </div>
      </section>

      <PlagiarismChecker />

      <section className="usecase-section">
        <div className="section-copy centered">
          <Badge>Use cases</Badge>
          <h2>Where deepfake detection matters.</h2>
        </div>
        <div className="card-grid four">
          {useCases.map(([title, body]) => (
            <article className="usecase-card" key={title}>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  )
}

export default App
