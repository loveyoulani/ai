// server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

dotenv.config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI)
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => console.error('MongoDB connection error:', err));

// Models
const userSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true },
    password: { type: String, required: true }
});

const sessionSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    name: String,
    createdAt: { type: Date, default: Date.now }
});

const messageSchema = new mongoose.Schema({
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Session', required: true },
    role: { type: String, enum: ['user', 'assistant'], required: true },
    content: { type: String, required: true },
    timestamp: { type: Date, default: Date.now }
});

const User = mongoose.model('User', userSchema);
const Session = mongoose.model('Session', sessionSchema);
const Message = mongoose.model('Message', messageSchema);

// Authentication Middleware
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) return res.status(401).json({ message: 'No token provided' });

    jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
        if (err) return res.status(403).json({ message: 'Invalid token' });
        req.user = user;
        next();
    });
};

// Auth Routes
app.post('/api/register', async (req, res) => {
    try {
        const { username, password } = req.body;

        // Check if user exists
        const existingUser = await User.findOne({ username });
        if (existingUser) {
            return res.status(400).json({ message: 'Username already exists' });
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create user
        const user = new User({
            username,
            password: hashedPassword
        });

        await user.save();

        // Generate token
        const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '24h' });

        res.status(201).json({ token });
    } catch (error) {
        res.status(500).json({ message: 'Error creating user' });
    }
});

app.post('/api/login', async (req, res) => {
    try {
        const { username, password } = req.body;

        // Find user
        const user = await User.findOne({ username });
        if (!user) {
            return res.status(400).json({ message: 'User not found' });
        }

        // Check password
        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            return res.status(400).json({ message: 'Invalid password' });
        }

        // Generate token
        const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '24h' });

        res.json({ token });
    } catch (error) {
        res.status(500).json({ message: 'Error logging in' });
    }
});

// Session Routes
app.get('/api/sessions', authenticateToken, async (req, res) => {
    try {
        const sessions = await Session.find({ userId: req.user.id })
            .sort({ createdAt: -1 });
        res.json(sessions);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching sessions' });
    }
});

app.post('/api/sessions', authenticateToken, async (req, res) => {
    try {
        const session = new Session({
            userId: req.user.id,
            name: req.body.name || `Chat ${Date.now()}`
        });
        await session.save();
        res.status(201).json(session);
    } catch (error) {
        res.status(500).json({ message: 'Error creating session' });
    }
});

// Message Routes
app.get('/api/messages/:sessionId', authenticateToken, async (req, res) => {
    try {
        const session = await Session.findOne({
            _id: req.params.sessionId,
            userId: req.user.id
        });

        if (!session) {
            return res.status(404).json({ message: 'Session not found' });
        }

        const messages = await Message.find({ sessionId: req.params.sessionId })
            .sort({ timestamp: 1 });
        res.json(messages);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching messages' });
    }
});

// Chat Route
app.post('/api/chat', authenticateToken, async (req, res) => {
    try {
        const { sessionId, message } = req.body;

        // Verify session belongs to user
        const session = await Session.findOne({
            _id: sessionId,
            userId: req.user.id
        });

        if (!session) {
            return res.status(404).json({ message: 'Session not found' });
        }

        // Save user message
        const userMessage = new Message({
            sessionId,
            role: 'user',
            content: message
        });
        await userMessage.save();

        // Initialize Groq client
        const groq = new Groq(process.env.GROQ_API_KEY);

        // Get response from Groq
        const completion = await groq.chat.completions.create({
            messages: [{ role: 'user', content: message }],
            model: 'mixtral-8x7b-32768',
            temperature: 0.7
        });

        const assistantResponse = completion.choices[0].message.content;

        // Save assistant message
        const assistantMessage = new Message({
            sessionId,
            role: 'assistant',
            content: assistantResponse
        });
        await assistantMessage.save();

        res.json({ response: assistantResponse });
    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ message: 'Error processing chat message' });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ message: 'Something broke!' });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
