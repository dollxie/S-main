# SceneSolver - Crime Detection System

A real-time crime detection system that uses AI to analyze video footage and identify potential criminal activities.

## Features

- Video upload and analysis
- Real-time crime detection using Timesformer and CLIP models
- Object detection for suspicious items
- AI-powered video captioning
- Evidence extraction and documentation

## Tech Stack

- Backend: Python, Flask
- Frontend: React, Material-UI
- AI Models: Timesformer, CLIP, YOLOv8
- Deployment: Vercel

## Setup Instructions

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/scenesolver.git
cd scenesolver
```

2. Set up the backend:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Create a `.env` file in the root directory with your API keys:
```
GOOGLE_API_KEY=your_google_api_key
```

5. Start the development servers:
```bash
# Terminal 1 (Backend)
python app.py

# Terminal 2 (Frontend)
cd frontend
npm start
```

### Deployment

1. Push your code to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. Connect your GitHub repository to Vercel:
   - Go to [Vercel](https://vercel.com)
   - Import your repository
   - Configure the build settings:
     - Framework Preset: Other
     - Build Command: `cd frontend && npm install && npm run build`
     - Output Directory: `frontend/build`
   - Add environment variables in Vercel dashboard

3. Deploy:
   - Vercel will automatically deploy your application
   - You can also trigger manual deployments from the Vercel dashboard

## Environment Variables

- `GOOGLE_API_KEY`: Your Google API key for Gemini model
- `FLASK_ENV`: Set to 'production' for deployment
- `FLASK_APP`: Set to 'app.py'

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 