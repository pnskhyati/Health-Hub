const { OpenAI } = require('openai');
const express = require('express');
const router = express.Router();

// Initialize OpenAI
const openai = new OpenAI({
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: process.env.OPENROUTER_API_KEY,
});

// Nutrition Chat Endpoint
router.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: "Message is required" });
        }

        const response = await openai.chat.completions.create({
            model: "deepseek/deepseek-r1-0528:free",
            messages: [
                {
                    role: "system",
                    content: `You are a professional nutritionist. Provide accurate, practical nutrition advice and meal plans. 
                    Follow these rules:
                    1. Be specific with portion sizes and measurements
                    2. Suggest balanced meals with proteins, carbs, and fats
                    3. Consider common dietary restrictions
                    4. Format responses clearly with line breaks for readability
                    5. If asked about medical conditions, recommend consulting a doctor`
                },
                { role: "user", content: message }
            ],
            temperature: 0.7,
            max_tokens: 1000
        });

        if (!response.choices || !response.choices[0] || !response.choices[0].message) {
            throw new Error("Invalid response structure from AI");
        }

        const reply = response.choices[0].message.content;
        res.json({ reply });
    } catch (error) {
        console.error('Error in /chat endpoint:', {
            message: error.message,
            stack: error.stack,
            response: error.response?.data
        });
        res.status(500).json({ 
            error: "Failed to process your request",
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
});

module.exports = router;