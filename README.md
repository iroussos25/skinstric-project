# Skinstric: AI Biometric Capture Interface

An AI-powered selfie capture module built during my software engineering internship at Skinstric (Jan-Feb 2026). Uses MediaPipe BlazeFace for real-time face detection with a custom 36-segment interactive guidance system.

**[Live Demo](https://skinstric-project-mu.vercel.app/)**

---

## What it does

Guides users through capturing a high-quality facial image for skin analysis. The interface provides real-time feedback on face positioning and alignment before accepting a capture.

## What I built beyond the original spec

The initial Figma design called for a basic capture screen. I extended it significantly:

1. **First attempt - pixel-based detection:** Used `requestAnimationFrame` with pixel change analysis to detect face presence. The oval turned green for "face detected" - but this was binary (face/no face) and prone to false positives from hand movements or shadows.

2. **Upgrade to BlazeFace:** Researched professional face detection approaches and implemented MediaPipe BlazeFace for sub-100ms client-side face landmark detection. No server round-trip needed.

3. **36-segment guidance oval:** Instead of a simple green/gray binary indicator, I built an interactive oval divided into 36 segments. Each segment lights up individually based on how well the user's face aligns with that region. When all segments are green, the UI displays "Perfect! Hold still!" and captures automatically.

## Key technical details

- MediaPipe BlazeFace for real-time 3D facial landmark detection (runs entirely client-side)
- Custom 36-segment SVG overlay with per-segment state management
- React state handling for real-time feedback loop (detection > evaluation > UI update)
- Browser Media API for camera access and stream management
- Secure image upload pipeline to Firebase Cloud Storage via Next.js
- Health tracking dashboard for clinical data visualization

## Run locally
```bash
git clone https://github.com/iroussos25/skinstric-project.git
cd skinstric-project
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) - requires camera access.

## Tech stack

TypeScript, React, Next.js, MediaPipe BlazeFace, Browser Media API, Firebase (Cloud Storage, Firestore), Tailwind CSS, Vercel

## Context

This was an internship project at Skinstric (Jan-Feb 2026). The core requirement was a selfie capture module for a skincare analysis pipeline. I extended the scope to include real-time ML-based face detection and the segment-by-segment guidance system, which were not part of the original brief.

## Contact

Giannis Roussos - [giannisroussos.com](https://giannisroussos.com) | [LinkedIn](https://linkedin.com/in/giannisr) | grcodes@outlook.com
