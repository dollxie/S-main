# SceneSolver - Crime Detection System

SceneSolver is a real-time AI-powered crime detection system that analyzes video footage to identify potential criminal activities using advanced computer vision models.

---

## Features

- Video upload and analysis interface  
- Real-time crime classification using TimeSformer and CLIP  
- Suspicious object detection via YOLOv8  
- AI-powered video captioning for forensic insights  
- Automatic evidence extraction and documentation  

---

## Tech Stack

| Layer        | Tools & Frameworks             |
|--------------|-------------------------------|
| Frontend     | React.js, Material UI         |
| Backend      | Python, Flask                 |
| AI Models    | TimeSformer, CLIP, YOLOv8     |
| Deployment   | Vercel |

---

## Setup Instructions

### Local Development

1. Clone the repository
 ```bash
   git clone https://github.com/yourusername/scenesolver.git
   cd scenesolver
```

2. Set up the backend:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Add environment variables:
```
GOOGLE_API_KEY=your_google_api_key
FLASK_ENV=development
FLASK_APP=app.py
```

5. Run the development servers:
- Terminal 1 (Backend)
```bash
python app.py
```
- Terminal 2 (Frontend)
```bash
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

2. Connect to Vercel:
   - Go to [Vercel](https://vercel.com)
   - Import your repository
   - Configure the build settings:
     - Framework Preset: Other
     - Build Command: `cd frontend && npm install && npm run build`
     - Output Directory: `frontend/build`
   - Add environment variables in Vercel dashboard

3. Deploy:
   - Vercel will automatically deploy your app on push
   - Manual deployments can also be triggered from the dashboard

## Environment Variables

| Variable         | Description                             |
|------------------|-----------------------------------------|
| `GOOGLE_API_KEY` | API key for Google Gemini model         |
| `FLASK_ENV`      | Set to `production` or `development`    |
| `FLASK_APP`      | Entry point for Flask backend (e.g. `app.py`) |


## Contribution

We welcome contributions to SceneSolver! To get started:

1. **Fork** the repository  
2. **Create a feature branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. Make your changes and commit them with a clear message
   ```bash
   git commit -m "Add: Short description of your feature"
   ```
4. Push the branch to your fork
   ```bash
   git push origin feature/YourFeatureName
   ```
5. Open a Pull Request on the main repository
   - Clearly describe your changes
   - Reference any related issues

> **Note:**  
> This project is still in active development. Features, models, and APIs may change frequently.  
> If you encounter any issues or have suggestions, feel free to open an issue or contribute directly.


## License

This project is licensed under the MIT License - see the LICENSE file for details. 
