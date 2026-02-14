import React, { useState, useRef, useEffect } from 'react'; // Import React tools for building the interface
import axios from 'axios'; // Import tool for talking to the backend server
import { Upload, Mic, Send, Loader2, Play, Volume2, X, FileText, CheckCircle2 } from 'lucide-react'; // Import icons
import { motion, AnimatePresence } from 'framer-motion'; // Import tool for smooth animations
import './App.css'; // Import the design styles

const App = () => { // Define the main frontend application
    // Variables that keep track of what's happening on the screen
    const [file, setFile] = useState(null); // Current uploaded PDF file
    const [isProcessing, setIsProcessing] = useState(false); // Is the AI currently reading the PDF?
    const [isReady, setIsReady] = useState(false); // Is the tutor ready to talk?
    const [messages, setMessages] = useState([]); // List of all chat messages
    const [inputText, setInputText] = useState(''); // What the user is currently typing
    const [isRecording, setIsRecording] = useState(false); // Is the microphone ON?
    const [isThinking, setIsThinking] = useState(false); // Is the AI brain working on an answer?
    const [isHandsFree, setIsHandsFree] = useState(false); // Is the "automatic" listening mode ON?
    const [isSpeaking, setIsSpeaking] = useState(false); // Is the AI currently talking out loud?
    const [isMuted, setIsMuted] = useState(false); // Is the AI voice muted?

    // References to hardware and special tools
    const mediaRecorderRef = useRef(null); // Tool for recording audio from mic
    const audioChunksRef = useRef([]); // Temporary storage for recorded sound data
    const chatEndRef = useRef(null); // Reference to the bottom of the chat list
    const audioRef = useRef(new Audio()); // The sound player for AI voice
    const silenceTimeoutRef = useRef(null); // Timer for automatic submission (hands-free)
    const audioContextRef = useRef(null); // Complex audio tool for measuring volume
    const analyserRef = useRef(null); // Tool that detects if user is being quiet
    const recognitionRef = useRef(null); // Browser's built-in tool for live text preview



    // Automatically scroll to the bottom when new messages arrive
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isThinking]);

    // Setup browser's built-in voice-to-text for real-time preview (live typing)
    useEffect(() => {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = true;
            recognitionRef.current.interimResults = true;

            recognitionRef.current.onresult = (event) => {
                let interimTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (!event.results[i].isFinal) {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                if (interimTranscript) setInputText(interimTranscript); // Show text as user speaks
            };
        }
    }, []);

    // Watch for "Hands-Free" mode and start listening automatically if ready
    useEffect(() => {
        if (isHandsFree && !isSpeaking && !isThinking && !isProcessing && isReady && !isRecording) {
            startRecording();
        }
    }, [isHandsFree, isSpeaking, isThinking, isProcessing, isReady, isRecording]);

    // Function to stop the AI from talking
    const stopAudio = () => {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        setIsSpeaking(false);
    };

    // Function to return to the upload screen
    const handleBack = () => {
        setIsReady(false);
        setFile(null);
        setMessages([]);
        stopAudio();
    };

    // Function to handle PDF file selection and upload
    const handleFileUpload = async (e) => {
        const uploadedFile = e.target.files[0];
        if (!uploadedFile || uploadedFile.type !== 'application/pdf') {
            alert('Please upload a valid PDF file.');
            return;
        }

        setFile(uploadedFile); // Set the file
        setIsProcessing(true); // Show the loading screen

        const formData = new FormData();
        formData.append('file', uploadedFile); // Put file in a delivery box
        formData.append('rebuild_index', 'true'); // Tell server to make a new DB

        try {
            const response = await axios.post('/api/upload', formData); // Send to backend
            setIsProcessing(false); // Stop loading screen
            setIsReady(true); // Show the chat screen

            // Play a greeting if the AI says hello first
            if (response.data.greeting_audio) {
                const audioPath = `/audio/${response.data.greeting_audio.split('\\').pop().split('/').pop()}`;
                audioRef.current.src = audioPath;
                setIsSpeaking(true);
                audioRef.current.play();
                audioRef.current.onended = () => setIsSpeaking(false);
            }
        } catch (error) {
            console.error('Upload failed:', error);
            setIsProcessing(false);

            // Handle specific errors for users
            const errorDetail = error.response?.data?.detail || error.message;
            let userMessage = 'Failed to process document.';

            if (errorDetail.includes('ERROR_POPPLER_MISSING')) {
                userMessage = '❌ Poppler Missing: Scanned PDF requires Poppler tools.';
            } else if (errorDetail.includes('ERROR_TESSERACT_MISSING')) {
                userMessage = '❌ Tesseract Missing: Scanned PDF requires OCR tools.';
            } else {
                userMessage = `❌ Error: ${errorDetail}`;
            }

            alert(userMessage);
        }
    };

    // Function to start capturing sound from the microphone
    const startRecording = async () => {
        if (isSpeaking) stopAudio(); // Silence AI if it was talking

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); // Request mic access
            mediaRecorderRef.current = new MediaRecorder(stream); // Setup the recorder
            audioChunksRef.current = []; // Fresh list for sound data

            // "Listen" for silence if hands-free is ON
            if (isHandsFree) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
                analyserRef.current = audioContextRef.current.createAnalyser();
                const source = audioContextRef.current.createMediaStreamSource(stream);
                source.connect(analyserRef.current);
                analyserRef.current.fftSize = 256;

                const bufferLength = analyserRef.current.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);

                const checkSilence = () => {
                    if (!analyserRef.current || !navigator.mediaDevices) return;
                    analyserRef.current.getByteFrequencyData(dataArray);
                    const average = dataArray.reduce((a, b) => a + b) / bufferLength;

                    if (average < 15) { // If sound level is very low
                        if (!silenceTimeoutRef.current) {
                            silenceTimeoutRef.current = setTimeout(() => {
                                stopRecording(); // Automatically submit after 6 seconds of silence
                            }, 6000);
                        }
                    } else {
                        if (silenceTimeoutRef.current) { // If user started talking again
                            clearTimeout(silenceTimeoutRef.current); // Reset the timer
                            silenceTimeoutRef.current = null;
                        }
                    }
                    if (isRecording) requestAnimationFrame(checkSilence); // Keep checking
                };
                checkSilence();
            }

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunksRef.current.push(e.data); // Save sound data pieces
            };

            mediaRecorderRef.current.onstop = async () => { // When recording stops
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' }); // Combine pieces into a file
                if (audioChunksRef.current.length > 0) {
                    await sendAudioQuestion(audioBlob); // Send voice file to server
                }
                if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
                    audioContextRef.current.close();
                }
            };

            mediaRecorderRef.current.start(); // Start now
            setIsRecording(true);

            if (recognitionRef.current) {
                try {
                    recognitionRef.current.start(); // Start live text preview
                } catch (e) {
                    console.warn("Recognition already started");
                }
            }
        } catch (error) {
            console.error('Error accessing microphone:', error);
            if (!isHandsFree) alert('Please allow microphone access.');
        }
    };

    // Function to stop the microphone
    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop()); // Turn off mic light
            if (recognitionRef.current) {
                try {
                    recognitionRef.current.stop();
                } catch (e) { }
            }
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
                silenceTimeoutRef.current = null;
            }
        }
    };

    // Function to send a typed question to the AI server
    const sendTextQuestion = async () => {
        if (!inputText.trim()) return;
        if (isSpeaking) stopAudio();

        const question = inputText;
        setInputText('');
        setMessages(prev => [...prev, { role: 'user', content: question }]); // Show user message
        setIsThinking(true); // Show "typing" bubbles

        try {
            const formData = new FormData();
            formData.append('text', question);
            formData.append('use_retrieval', 'true');
            formData.append('return_audio', 'true');

            const response = await axios.post('/api/ask', formData); // Talk to backend
            handleAgentResponse(response.data); // Handle the AI answer
        } catch (error) {
            console.error('Question failed:', error);
            setIsThinking(false);
        }
    };

    // Function to send a voice recording to the AI server
    const sendAudioQuestion = async (audioBlob) => {
        setIsThinking(true);

        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'question.wav');
            formData.append('use_retrieval', 'true');
            formData.append('return_audio', 'true');

            const response = await axios.post('/api/ask', formData);

            if (response.data.question) {
                setMessages(prev => [...prev, { role: 'user', content: response.data.question }]);
                handleAgentResponse(response.data);
            } else {
                setIsThinking(false);
            }
        } catch (error) {
            console.error('Audio question failed:', error);
            setIsThinking(false);
        }
    };

    // Function to process what the AI sent back (text + voice)
    const handleAgentResponse = (data) => {
        setIsThinking(false);

        const fullAnswer = data.answer;

        // Add an empty assistant message first
        setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

        const startTypewriter = (duration) => {
            let currentIndex = 0;
            const textLength = fullAnswer.length;
            // Calculate how long to wait between each character
            // duration is in seconds, so we convert to ms and divide by length
            const totalMs = duration * 1000;
            const intervalTime = totalMs / textLength;

            const typingInterval = setInterval(() => {
                if (currentIndex <= textLength) {
                    const partialText = fullAnswer.slice(0, currentIndex);
                    setMessages(prev => {
                        const newMessages = [...prev];
                        if (newMessages.length > 0 && newMessages[newMessages.length - 1].role === 'assistant') {
                            newMessages[newMessages.length - 1].content = partialText;
                        }
                        return newMessages;
                    });
                    currentIndex++;
                } else {
                    clearInterval(typingInterval);
                }
            }, intervalTime);
        };

        if (data.audio_path && !isMuted) {
            const audioPath = `/audio/${data.audio_path.split('\\').pop().split('/').pop()}`;
            audioRef.current.src = audioPath;

            // Wait for audio to be ready so we know its duration
            audioRef.current.onloadedmetadata = () => {
                const duration = audioRef.current.duration;
                setIsSpeaking(true);
                audioRef.current.play();
                startTypewriter(duration);
            };

            audioRef.current.onended = () => setIsSpeaking(false);
        } else {
            // If muted or no audio, use a default reading speed (approx 50ms per char)
            startTypewriter(fullAnswer.length * 0.05);
        }
    };

    // Function to delete the conversation history
    const handleDeleteHistory = async () => {
        if (window.confirm('Are you sure you want to clear your conversation history?')) {
            try {
                await axios.post('/api/clear-memory');
                setMessages([]);
                stopAudio();
                alert('History cleared successfully.');
            } catch (error) {
                console.error('Failed to clear history:', error);
                alert('Failed to clear history on server.');
            }
        }
    };

    return ( // The actual HTML structure (JSX)
        <div className="app-container">

            <header className="app-header">
                <div className="header-left">
                    {isReady && (
                        <button className="back-btn" onClick={handleBack} title="Back to Upload">
                            <Play style={{ transform: 'rotate(180deg)' }} size={20} />
                            <span>Back</span>
                        </button>
                    )}
                </div>

                <div className="logo center-logo">
                    <FileText className="logo-icon" />
                    <span>ECHOLEARNER AI</span>
                    <span style={{ fontSize: '0.7rem', opacity: 0.5, marginLeft: '5px', verticalAlign: 'top' }}>v2.0</span>
                </div>

                <div className="header-right">
                    <div className="status">
                        {isReady ? (
                            <div className="status-tag ready">
                                <CheckCircle2 size={16} />
                                <span>Ready</span>
                            </div>
                        ) : (
                            <div className="status-tag wait">
                                <Loader2 size={16} className="spin" />
                                <span>Waiting</span>
                            </div>
                        )}
                    </div>
                </div>
            </header>

            <main className="main-content">
                <AnimatePresence mode="wait">
                    {!isReady && !isProcessing ? ( // Welcome screen with upload button
                        <motion.div
                            key="upload"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="upload-zone"
                        >
                            <div className="upload-card">
                                <div className="icon-wrapper">
                                    <Upload size={48} />
                                </div>
                                <h2>Analyze Your PDF</h2>
                                <p>Upload a standard or scanned PDF to start a real-time learning session with your AI tutor.</p>
                                <label className="upload-button">
                                    Choose PDF
                                    <input type="file" accept=".pdf" onChange={handleFileUpload} hidden />
                                </label>
                            </div>
                        </motion.div>
                    ) : isProcessing ? ( // Loading screen while reading PDF
                        <motion.div
                            key="processing"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="processing-zone"
                        >
                            <div className="loader-wrapper">
                                <Loader2 size={64} className="spin primary" />
                                <h3>Analyzing Document...</h3>
                                <p>Running OCR and semantic indexing to prepare your tutor.</p>
                            </div>
                        </motion.div>
                    ) : ( // Active chat interface
                        <motion.div
                            key="chat"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="chat-zone"
                        >
                            <div className="chat-window-bordered">
                                <div className="chat-window-header">
                                    <div className="ai-avatar-board">
                                        <div className="avatar-glow"></div>
                                        <img src="/assets/robot_avatar.png" alt="AI Agent" className="robot-img" />
                                        <div className="avatar-info">
                                            <h3>Echo AI Tutor</h3>
                                            <div className="pulse-indicator">
                                                <div className="pulse-dot"></div>
                                                <span>Active</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="chat-header-actions">
                                        <button
                                            className={`hands-free-toggle ${isHandsFree ? 'active' : ''}`}
                                            onClick={() => setIsHandsFree(!isHandsFree)}
                                            title="Continuous conversation mode"
                                        >
                                            <div className="toggle-switch">
                                                <div className="toggle-dot"></div>
                                            </div>
                                            <span>Hands-Free</span>
                                        </button>

                                        {isSpeaking && (
                                            <button className="stop-btn" onClick={stopAudio} title="Stop Agent">
                                                <X size={20} />
                                            </button>
                                        )}

                                        <button
                                            className={`mute-toggle ${isMuted ? 'active' : ''}`}
                                            onClick={() => {
                                                if (!isMuted) stopAudio();
                                                setIsMuted(!isMuted);
                                            }}
                                            title={isMuted ? "Unmute" : "Mute AI Voice"}
                                        >
                                            <Volume2 size={20} />
                                            <span>{isMuted ? "Muted" : "Mute"}</span>
                                        </button>

                                        {isReady && (
                                            <button className="delete-btn-inline" onClick={handleDeleteHistory} title="Clear Conversation">
                                                <X size={18} />
                                                <span>Clear Chat</span>
                                            </button>
                                        )}
                                    </div>
                                </div>

                                <div className="chat-messages">

                                    {messages.map((msg, idx) => ( // Loop through and show all chat history
                                        <div key={idx} className={`message ${msg.role}`}>
                                            <div className="message-content">
                                                <p>{msg.content}</p>
                                            </div>
                                        </div>
                                    ))}
                                    {isThinking && ( // Show bubbles when AI is thinking
                                        <div className="message assistant">
                                            <div className="message-content thinking">
                                                <div className="dot"></div>
                                                <div className="dot"></div>
                                                <div className="dot"></div>
                                            </div>
                                        </div>
                                    )}
                                    <div ref={chatEndRef} />
                                </div>

                                <div className="input-zone">
                                    <div className="input-wrapper">
                                        <button // Pulse mic button
                                            className={`voice-btn ${isRecording ? 'recording' : ''}`}
                                            onMouseDown={!isHandsFree ? startRecording : null}
                                            onMouseUp={!isHandsFree ? stopRecording : null}
                                            onClick={isHandsFree ? () => setIsHandsFree(!isHandsFree) : null}
                                        >
                                            {isRecording ? <div className="pulse" /> : <Mic />}
                                        </button>
                                        <textarea
                                            placeholder={isHandsFree ? "Listening automatically..." : "Type your question..."}
                                            value={inputText}
                                            onChange={(e) => setInputText(e.target.value)}
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter' && !e.shiftKey) {
                                                    e.preventDefault();
                                                    sendTextQuestion();
                                                }
                                            }}
                                            disabled={isHandsFree && isRecording}
                                            rows="1"
                                        />
                                        <button className="send-btn" onClick={sendTextQuestion}>
                                            <Send size={20} />
                                        </button>
                                    </div>


                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </main>
        </div>
    );
};

export default App; // Export the whole app
