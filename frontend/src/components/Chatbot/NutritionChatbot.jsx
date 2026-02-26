import React, { useState, useRef, useEffect } from 'react';
import './NutritionChatbot.css';

const NutritionChatbot = () => {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! I can help with nutrition advice and meal plans. Ask me anything!' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || loading) return;
  
    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
  
    try {
      // Add timestamp to prevent caching issues
      const apiUrl = new URL('http://localhost:5000/api/chat');
      apiUrl.searchParams.append('_', Date.now());
  
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
        // credentials: 'include' // Only needed if using cookies/sessions
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json();
      setMessages(prev => [...prev, { sender: 'bot', text: data.reply }]);
    } catch (error) {
      console.error('Fetch error:', error);
      setMessages(prev => [...prev, { 
        sender: 'bot', 
        text: `Network error: ${error.message}. Please ensure:
        1. Backend server is running (node server.js)
        2. You're using the correct URL
        3. No ad blockers are interfering`
      }]);
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h2>Nutrition Expert</h2>
      </div>
      
      <div className="chatbot-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.sender}`}>
            {msg.text.split('\n').map((paragraph, idx) => (
              <p key={idx}>{paragraph}</p>
            ))}
          </div>
        ))}
        {loading && <div className="message bot">Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chatbot-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Ask about nutrition..."
          disabled={loading}
        />
        <button 
          onClick={handleSendMessage}
          disabled={loading || !input.trim()}
        >
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default NutritionChatbot;