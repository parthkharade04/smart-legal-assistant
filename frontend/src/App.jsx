import ReactMarkdown from 'react-markdown'
import { useState, useRef, useEffect } from 'react'
import './index.css'

// Helper Component for Highlighting Text
const HighlightedText = ({ text, highlight }) => {
  if (!highlight.trim()) return <span>{text}</span>

  // Clean special chars from regex
  const safeHighlight = highlight.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${safeHighlight})`, 'gi'))

  return (
    <span>
      {parts.map((part, i) =>
        part.toLowerCase() === highlight.toLowerCase() ?
          <span key={i} className="highlight">{part}</span> : part
      )}
    </span>
  )
}

function App() {
  const [query, setQuery] = useState('')
  const [lastQuery, setLastQuery] = useState('')
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello! I am your AI Legal Assistant. I have analyzed the contracts in your workspace. Ask me anything about them.' }
  ])
  const [sources, setSources] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [selectedDocContent, setSelectedDocContent] = useState('')
  const [selectedDocTitle, setSelectedDocTitle] = useState('')
  const [isDocLoading, setIsDocLoading] = useState(false)

  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Dynamic API URL (Load from Env or default to localhost)
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    const userMsg = { role: 'user', text: query }
    setMessages(prev => [...prev, userMsg])
    setLastQuery(query) // Save for highlighting
    setQuery('')
    setIsLoading(true)
    setSources([])

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.text })
      })

      const data = await response.json()
      setMessages(prev => [...prev, { role: 'bot', text: data.answer }])
      setSources(data.sources)
    } catch (error) {
      console.error(error)
      setMessages(prev => [...prev, { role: 'bot', text: "Sorry, I encountered an error connecting to the legal engine." }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleViewDocument = async (filename) => {
    setIsModalOpen(true)
    setSelectedDocTitle(filename)
    setSelectedDocContent('')
    setIsDocLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/document/${filename}`)
      if (!response.ok) throw new Error("Failed to load")
      const data = await response.json()
      setSelectedDocContent(data.content)
    } catch (err) {
      setSelectedDocContent("Error loading full document content.")
    } finally {
      setIsDocLoading(false)
    }
  }

  return (
    <div className="app-container">
      {/* Left Panel: Evidence / Sources */}
      <div className="left-panel">
        <h1>📑 Contract Intelligence</h1>
        <h2>Relevant Clauses</h2>

        <div className="sources-list">
          {(sources || []).length === 0 ? (
            <div style={{ opacity: 0.5, fontStyle: 'italic' }}>
              Evidence retrieval will appear here when you ask a question.
            </div>
          ) : (
            (sources || []).map((src, idx) => (
              <div key={idx} className="source-card">
                <span className="source-title">SOURCE: {src.source}</span>
                <div className="source-text">
                  "<HighlightedText text={src.text} highlight={lastQuery} />"
                </div>
                <button
                  className="view-doc-btn"
                  onClick={() => handleViewDocument(src.source)}
                >
                  View Full Contract
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Right Panel: Chat */}
      <div className="right-panel">
        <h1>⚖️ Legal Assistant</h1>

        <div className="chat-window">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              {msg.role === 'bot' ? (
                <div className="markdown-body">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              ) : (
                <p>{msg.text}</p>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="message bot">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className="input-area" onSubmit={handleSearch}>
          <input
            type="text"
            placeholder="Ask a question about the contracts..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Wait' : 'Ask'}
          </button>
        </form>
      </div>

      {/* Full Document Modal */}
      {isModalOpen && (
        <div className="modal-overlay" onClick={() => setIsModalOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>📄 {selectedDocTitle}</h2>
              <button className="close-button" onClick={() => setIsModalOpen(false)}>×</button>
            </div>
            <div className="modal-body">
              {isDocLoading ? (
                <div className="typing-indicator"><span></span><span></span><span></span></div>
              ) : (
                <HighlightedText text={selectedDocContent} highlight={lastQuery} />
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

export default App
