# Topic Modeling Demo

A live, full-stack demo of the topic-modeling project: a FastAPI backend
running LDA/NMF topic modeling, and a React frontend to submit text and
see the results.

```
ml-topic-demo/
  backend/     FastAPI app (Python)
  frontend/    React app (Vite)
```

## 1. Run it locally first

**Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Visit http://localhost:8000/docs to try the API directly.

**Frontend** (in a new terminal)
```bash
cd frontend
npm install
cp .env.example .env          # VITE_API_URL=http://localhost:8000
npm run dev
```
Visit http://localhost:5173 — paste text, click "Find Topics".

## 2. Push to GitHub

Create one repo (e.g. `topic-modeling-demo`) containing both `backend/`
and `frontend/` folders, and push this whole project to it.

## 3. Deploy the backend — Render (free)

1. Go to render.com → sign up with GitHub (no card required).
2. **New +** → **Web Service** → connect your repo.
3. Set **Root Directory** to `backend`.
4. Render will detect the `Dockerfile` automatically — leave environment
   as Docker.
5. Instance type: **Free**.
6. Deploy. You'll get a URL like `https://topic-modeling-demo.onrender.com`.

Note: on the free tier the service sleeps after 15 minutes of no traffic,
so the first request after a while takes ~30-50s to wake up. Totally fine
for a portfolio demo — just don't be surprised by it in an interview.

## 4. Deploy the frontend — Vercel (free)

1. Go to vercel.com → sign up with GitHub.
2. **Add New** → **Project** → import the same repo.
3. Set **Root Directory** to `frontend`.
4. Framework preset: **Vite** (auto-detected).
5. Add an environment variable:
   `VITE_API_URL` = your Render backend URL from step 3.
6. Deploy. You'll get a URL like `https://topic-modeling-demo.vercel.app`.

## 5. Put it on your resume / GitHub

Once both are live, add the Vercel link as the "live demo" for this
project — e.g.:

```
Topic Modeling and Document Categorization | NLP, LDA, Gensim, Scikit-Learn
Live demo: https://topic-modeling-demo.vercel.app
```

A working link a recruiter can click beats a project bullet every time.
