"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

type CameraCaptureModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (base64Image: string) => Promise<void>;
};

const OVAL_SEGMENT_COUNT = 36;

export default function CameraCaptureModal({
  isOpen,
  onClose,
  onCapture,
}: CameraCaptureModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectionCanvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [timerSetting, setTimerSetting] = useState<"OFF" | "3s" | "10s">("OFF");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isFaceDetected, setIsFaceDetected] = useState(false);
  const [isPerfectlyPlaced, setIsPerfectlyPlaced] = useState(false);
  const [segmentConfidences, setSegmentConfidences] = useState<number[]>(() =>
    Array(OVAL_SEGMENT_COUNT).fill(0)
  );
  const [showTimerOptions, setShowTimerOptions] = useState(false);
  const [faceDetectorReady, setFaceDetectorReady] = useState(false);
  const [faceDetectorLoading, setFaceDetectorLoading] = useState(true);
  const faceDetectorRef = useRef<FaceDetector | null>(null);
  const segmentConfidenceRef = useRef<number[]>(Array(OVAL_SEGMENT_COUNT).fill(0));

  useEffect(() => {
    if (!isOpen) {
      // Clean up stream when modal closes
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        setStream(null);
      }
      setCapturedImage(null);
      setError(null);
      setIsFaceDetected(false);
      setIsPerfectlyPlaced(false);
      setSegmentConfidences(Array(OVAL_SEGMENT_COUNT).fill(0));
      segmentConfidenceRef.current = Array(OVAL_SEGMENT_COUNT).fill(0);
      return;
    }

    // Stop existing stream before requesting a new one (e.g., when switching facingMode)
    const stopExistingStream = () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };

    const startCamera = async () => {
      try {
        setError(null); // Clear previous errors
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: facingMode },
          audio: false,
        });
        setStream(mediaStream);
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (err) {
        let errorMessage = "Failed to access camera. Please check permissions.";
        if (err instanceof DOMException) {
          if (err.name === "NotAllowedError" || err.message.includes("Permission denied")) {
            errorMessage = "Camera permission denied. Please enable camera access in your browser settings and try again.";
          } else if (err.name === "NotFoundError" || err.message.includes("NotFoundError")) {
            errorMessage = "No camera device found. Please check that a camera is connected.";
          } else if (err.name === "NotReadableError" || err.message.includes("NotReadableError")) {
            errorMessage = "Camera is already in use by another application. Please close other apps using the camera and try again.";
          }
        } else if (err instanceof Error) {
          errorMessage = err.message;
        }
        setError(errorMessage);
        setStream(null);
      }
    };

    // Stop old stream before starting new one
    stopExistingStream();
    
    // Add a small delay to ensure old stream is fully released
    const timerId = setTimeout(() => {
      startCamera();
    }, 100);

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    window.addEventListener("keydown", handleKeyDown);
    
    return () => {
      clearTimeout(timerId);
      document.body.style.overflow = originalOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      // Stream cleanup is handled in the main if (!isOpen) block above
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, onClose, facingMode]);

  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
    }
    
    return () => {
      // Stream cleanup is handled by the main camera effect
      // Just mark that we've cleaned up the display
    };
  }, [stream]);

  // Initialize MediaPipe FaceDetector
  useEffect(() => {
    let isMounted = true;
    
    const initFaceDetector = async () => {
      setFaceDetectorLoading(true);
      try {
        // Load MediaPipe vision WASM files from CDN
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        
        // Create face detector with video mode for real-time detection
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.5
        });
        
        if (isMounted) {
          faceDetectorRef.current = detector;
          setFaceDetectorReady(true);
          setFaceDetectorLoading(false);
          console.log("MediaPipe FaceDetector initialized successfully");
        }
      } catch (err) {
        console.error("MediaPipe FaceDetector initialization failed:", err);
        if (isMounted) {
          faceDetectorRef.current = null;
          setFaceDetectorReady(false);
          setFaceDetectorLoading(false);
        }
      }
    };

    initFaceDetector();
    
    return () => {
      isMounted = false;
      // Cleanup detector on unmount
      if (faceDetectorRef.current) {
        faceDetectorRef.current.close();
        faceDetectorRef.current = null;
      }
    };
  }, []);

  // Calculate which segments of the oval are overlapped by the face
  const calculateSegmentOverlap = useCallback(
    (
      faceBounds: { x: number; y: number; width: number; height: number },
      canvasWidth: number,
      canvasHeight: number,
      isMirrored: boolean
    ) => {
      const ovalCenterX = canvasWidth / 2;
      // Oval is positioned at 42% from top, not center
      const ovalCenterY = canvasHeight * 0.42;
      // Match the visual oval proportions relative to video
      // The oval takes roughly 50% of width and 70% of height visually
      const ovalRadiusX = canvasWidth * 0.25;
      const ovalRadiusY = canvasHeight * 0.35;

      // Mirror face bounds if camera is mirrored (front-facing)
      const adjustedFaceBounds = isMirrored
        ? {
            x: canvasWidth - faceBounds.x - faceBounds.width,
            y: faceBounds.y,
            width: faceBounds.width,
            height: faceBounds.height,
          }
        : faceBounds;

      const faceCenterX = adjustedFaceBounds.x + adjustedFaceBounds.width / 2;
      const faceCenterY = adjustedFaceBounds.y + adjustedFaceBounds.height / 2;
      const faceRadiusX = adjustedFaceBounds.width / 2;
      const faceRadiusY = adjustedFaceBounds.height / 2;

      const segmentAngle = (Math.PI * 2) / OVAL_SEGMENT_COUNT;
      const newSegmentValues: number[] = [];

      // Calculate how well the face is centered in the oval
      const centerOffsetX = Math.abs(faceCenterX - ovalCenterX) / ovalRadiusX;
      const centerOffsetY = Math.abs(faceCenterY - ovalCenterY) / ovalRadiusY;
      const centeringScore = Math.max(0, 1 - Math.sqrt(centerOffsetX * centerOffsetX + centerOffsetY * centerOffsetY));

      for (let i = 0; i < OVAL_SEGMENT_COUNT; i++) {
        // Start from top (-PI/2) and go clockwise
        const angle = -Math.PI / 2 + i * segmentAngle;
        
        // Point on oval perimeter
        const ovalPointX = ovalCenterX + Math.cos(angle) * ovalRadiusX;
        const ovalPointY = ovalCenterY + Math.sin(angle) * ovalRadiusY;
        
        // Corresponding point on face perimeter (scaled from face center)
        const facePointX = faceCenterX + Math.cos(angle) * faceRadiusX;
        const facePointY = faceCenterY + Math.sin(angle) * faceRadiusY;
        
        // Distance from face edge to oval edge at this angle
        const dx = ovalPointX - facePointX;
        const dy = ovalPointY - facePointY;
        const distToOval = Math.sqrt(dx * dx + dy * dy);
        
        // Normalize by oval size - smaller distance = better overlap
        const normalizedDist = distToOval / Math.max(ovalRadiusX, ovalRadiusY);
        
        // Segment lights up when face edge is close to or past oval edge
        // Also factor in centering for smoother feedback
        let confidence = 0;
        if (normalizedDist < 0.5) {
          // Face edge is very close to or past this segment
          confidence = 1.0;
        } else if (normalizedDist < 1.0) {
          // Face edge is approaching this segment
          confidence = 1.0 - (normalizedDist - 0.5) * 2;
        }
        
        // Boost confidence when face is well-centered
        confidence = confidence * (0.5 + centeringScore * 0.5);
        
        newSegmentValues.push(Math.min(1, confidence));
      }

      return newSegmentValues;
    },
    []
  );

  // Face detection using FaceDetector API ONLY (no fallback to avoid false positives)
  useEffect(() => {
    if (!stream || !videoRef.current || !detectionCanvasRef.current) return;

    let animationFrameId: number;
    let isDetecting = false;
    const video = videoRef.current;
    const canvas = detectionCanvasRef.current;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const isMirrored = facingMode === "user";

    if (!context) return;

    let lastDetectionTime = 0;
    
    const detectFace = () => {
      if (video.readyState !== video.HAVE_ENOUGH_DATA || isDetecting) {
        animationFrameId = requestAnimationFrame(detectFace);
        return;
      }

      isDetecting = true;
      
      // Get current timestamp for video mode detection
      const nowMs = performance.now();
      
      // Throttle detection to ~30fps for performance
      if (nowMs - lastDetectionTime < 33) {
        isDetecting = false;
        animationFrameId = requestAnimationFrame(detectFace);
        return;
      }
      lastDetectionTime = nowMs;

      let faceBounds: { x: number; y: number; width: number; height: number } | null = null;

      // Use MediaPipe FaceDetector
      if (faceDetectorRef.current) {
        try {
          const result = faceDetectorRef.current.detectForVideo(video, nowMs);
          if (result.detections.length > 0) {
            const detection = result.detections[0];
            const bb = detection.boundingBox;
            
            if (bb) {
              // Convert normalized coordinates to pixel values
              const videoWidth = video.videoWidth;
              const videoHeight = video.videoHeight;
              
              // Validate face size is reasonable (not too small/large)
              const faceArea = bb.width * bb.height;
              const videoArea = videoWidth * videoHeight;
              const faceRatio = faceArea / videoArea;
              
              if (faceRatio > 0.03 && faceRatio < 0.5) {
                faceBounds = { 
                  x: bb.originX, 
                  y: bb.originY, 
                  width: bb.width, 
                  height: bb.height 
                };
              }
            }
          }
        } catch {
          // FaceDetector failed - no face detected
        }
      }

      // Calculate segment overlaps with smooth interpolation
      const previousConfidence = segmentConfidenceRef.current;
      let nextConfidence: number[];
      const hasFaceThisFrame = faceBounds !== null;

      if (faceBounds) {
        const rawOverlap = calculateSegmentOverlap(faceBounds, video.videoWidth, video.videoHeight, isMirrored);

        // Smooth temporal interpolation for fluid animation
        nextConfidence = rawOverlap.map((value, index) => {
          const prev = previousConfidence[index] ?? 0;
          const target = Math.min(1, value);
          // Smooth transition rate
          const rate = target > prev ? 0.4 : 0.25;
          return prev + (target - prev) * rate;
        });

        // Light spatial smoothing for coherent appearance
        const smoothed = nextConfidence.map((_, index) => {
          const prevIdx = (index - 1 + OVAL_SEGMENT_COUNT) % OVAL_SEGMENT_COUNT;
          const nextIdx = (index + 1) % OVAL_SEGMENT_COUNT;
          return (
            nextConfidence[index] * 0.7 +
            nextConfidence[prevIdx] * 0.15 +
            nextConfidence[nextIdx] * 0.15
          );
        });
        nextConfidence = smoothed;
      } else {
        // Fast fade-out when no face detected - drop to 0 quickly
        nextConfidence = previousConfidence.map((prev) => prev * 0.7);
        // Zero out very small values to ensure clean state
        nextConfidence = nextConfidence.map((v) => (v < 0.05 ? 0 : v));
      }

      segmentConfidenceRef.current = nextConfidence;

      // Always update state for smooth animation
      setSegmentConfidences([...nextConfidence]);

      // Calculate overlap metrics
      const totalConf = nextConfidence.reduce((sum, c) => sum + c, 0);
      const overlap = totalConf / OVAL_SEGMENT_COUNT;
      const activeSegs = nextConfidence.filter((c) => c > 0.5).length;

      // Face is detected only if we have active segments AND found a face this frame
      const hasFace = hasFaceThisFrame && activeSegs >= 3;
      setIsFaceDetected(hasFace);

      // Perfect placement: must have face this frame + high overlap + most segments active
      const isPerfect = hasFaceThisFrame && overlap > 0.75 && activeSegs >= OVAL_SEGMENT_COUNT * 0.8;
      setIsPerfectlyPlaced(isPerfect);

      isDetecting = false;
      animationFrameId = requestAnimationFrame(detectFace);
    };

    detectFace();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [stream, facingMode, calculateSegmentOverlap]);

  // Safety cleanup: ensure stream is stopped if component unmounts
  useLayoutEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  const handleCaptureImmediate = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    if (!context) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const imageData = canvas.toDataURL("image/jpeg", 0.95);
    setCapturedImage(imageData);
  };

  // Countdown timer effect
  useEffect(() => {
    if (countdown === null || countdown === 0) return;

    const timer = setTimeout(() => {
      if (countdown === 1) {
        // Trigger capture when countdown reaches 0
        setCountdown(null);
        // Use a small delay to ensure countdown is hidden before capture
        setTimeout(() => {
          handleCaptureImmediate();
        }, 100);
      } else {
        setCountdown(countdown - 1);
      }
    }, 1000);

    return () => clearTimeout(timer);
  }, [countdown]);

  const handleCapture = () => {
    if (timerSetting === "OFF") {
      handleCaptureImmediate();
    } else {
      const seconds = timerSetting === "3s" ? 3 : 10;
      setCountdown(seconds);
    }
  };

  const handleRetake = () => {
    setCapturedImage(null);
  };

  const handleFlipCamera = async () => {
    // Stop current stream
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    // Toggle facing mode
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  const handleUpload = async () => {
    if (!capturedImage) return;

    setIsUploading(true);
    setError(null);

    try {
      // Extract base64 string (remove data:image/...;base64, prefix)
      const base64String = capturedImage.split(",")[1];
      await onCapture(base64String);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  if (!isOpen) {
    return null;
  }

  const totalConfidence = segmentConfidences.reduce((sum, c) => sum + c, 0);
  const overlapRatio = totalConfidence / OVAL_SEGMENT_COUNT;
  const segmentSpan = 360 / OVAL_SEGMENT_COUNT;

  // Generate smooth gradient - green where face overlaps, transparent elsewhere
  const borderGradient = `conic-gradient(from -90deg, ${segmentConfidences
    .map((confidence, index) => {
      const start = index * segmentSpan;
      const end = (index + 1) * segmentSpan;
      // Only show green where confidence > threshold, with smooth falloff
      const intensity = Math.max(0, (confidence - 0.15) / 0.85);
      const opacity = Math.min(1, intensity * 1.2);
      const color =
        isPerfectlyPlaced
          ? `rgba(74, 222, 128, ${opacity.toFixed(3)})` // Bright green when perfect
          : `rgba(74, 222, 128, ${(opacity * 0.9).toFixed(3)})`;
      return `${color} ${start}deg ${end}deg`;
    })
    .join(", ")})`;

  // Review screen (after capture)
  if (capturedImage) {
    return (
      <div
        className="absolute inset-0 z-50 flex items-center justify-center bg-black/40"
        onClick={onClose}
      >
        <div
          className="w-full max-w-md rounded-lg border border-[#E5E7EB] bg-white px-4 py-6 sm:px-6 md:px-8 text-[#1A1B1C] shadow-lg"
          onClick={(event) => event.stopPropagation()}
        >
          <p className="mb-4 text-sm uppercase tracking-[0.2em]">
            Great Shot!
          </p>

          <div className="space-y-4">
            {/* Error message */}
            {error && <p className="text-xs text-red-600">{error}</p>}

            {/* Captured image preview */}
            <div className="relative h-48 sm:h-64 md:h-80 overflow-hidden rounded-lg border border-[#E5E7EB] bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={capturedImage}
                alt="Captured"
                className="h-full w-full object-cover"
              />
            </div>

            {/* Action buttons */}
            <div className="flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={handleRetake}
                disabled={isUploading}
                className="cursor-pointer rounded-full border border-[#1A1B1C] px-4 py-2 text-xs uppercase tracking-[0.2em] disabled:opacity-50"
              >
                Retake
              </button>
              <button
                type="button"
                onClick={handleUpload}
                disabled={isUploading}
                className="cursor-pointer rounded-full bg-[#1A1B1C] px-4 py-2 text-xs uppercase tracking-[0.2em] text-white disabled:opacity-50"
              >
                {isUploading ? "Uploading..." : "Upload"}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Camera capture screen (full screen)
  return (
    <div 
      className="fixed inset-0 z-50 bg-black"
      onClick={() => showTimerOptions && setShowTimerOptions(false)}
    >
      {/* Video background */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="absolute inset-0 h-full w-full object-cover"
        style={{ transform: facingMode === "user" ? "scaleX(-1)" : "none" }}
      />
      <canvas ref={canvasRef} className="hidden" />
      <canvas ref={detectionCanvasRef} className="hidden" />

      {/* Header - Top Left */}
      <div className="absolute left-8 top-8">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-white">
          SKINSTRIC <span className="font-normal">[ ANALYSIS ]</span>
        </p>
      </div>

      {/* Flip camera button - Top Right */}
      <button
        type="button"
        onClick={handleFlipCamera}
        disabled={!stream || !!error}
        className="absolute right-8 top-8 cursor-pointer rounded-full bg-white/80 p-2 text-[#1A1B1C] backdrop-blur-sm transition hover:bg-white disabled:opacity-50"
        title="Flip camera"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 2v6h-6" />
          <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
          <path d="M3 22v-6h6" />
          <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
        </svg>
      </button>

      {/* Oval face guide - positioned slightly above center for natural face framing */}
      <div className="absolute left-1/2 top-[42%] -translate-x-1/2 -translate-y-1/2">
        <div
          className="relative h-120 w-89 rounded-[50%] sm:h-93.75 sm:w-69.5 md:h-140 md:w-104"
          style={{
            // Base white border
            border: "4px solid rgba(255, 255, 255, 0.4)",
            // Green glow intensifies with overlap, extra bright when perfect
            boxShadow: isPerfectlyPlaced
              ? "0 0 0 9999px rgba(0, 0, 0, 0.3), 0 0 50px rgba(74, 222, 128, 0.9), 0 0 80px rgba(74, 222, 128, 0.6)"
              : overlapRatio > 0.1
                ? `0 0 0 9999px rgba(0, 0, 0, 0.3), 0 0 ${15 + overlapRatio * 25}px rgba(74, 222, 128, ${Math.min(0.7, overlapRatio * 0.9)})`
                : "0 0 0 9999px rgba(0, 0, 0, 0.3)",
            transition: "box-shadow 0.2s ease-out",
          }}
        >
          {/* Green border overlay - colors only the segments where face overlaps */}
          <div
            className="pointer-events-none absolute rounded-[50%]"
            style={{
              // Position exactly on top of the border
              top: "-4px",
              left: "-4px",
              right: "-4px",
              bottom: "-4px",
              background: borderGradient,
              // Mask to show only the border ring (4px wide)
              mask: "radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 4px), #000 100%, transparent 100%)",
              WebkitMask: "radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 4px), #000 100%, transparent 100%)",
              borderRadius: "50%",
            }}
          />
          {/* Countdown dial - Top of oval */}
          {countdown !== null && countdown > 0 && (
            <div className="absolute left-1/2 top-8 -translate-x-1/2">
              <div className="relative flex h-24 w-32 items-center justify-center overflow-hidden">
                {/* Dial effect with rotating numbers */}
                <div className="absolute inset-0 flex items-center justify-center">
                  {/* Previous number (appears to rotate from right) */}
                  {countdown < 10 && (
                    <div
                      className="absolute text-4xl font-bold text-white transition-all duration-1000"
                      style={{
                        transform: `translateX(${countdown === (countdown + 1) ? '0px' : '70px'}) translateY(-10px) scale(0.6) rotateY(45deg)`,
                        opacity: 0.2,
                        filter: "blur(1px)",
                      }}
                    >
                      {countdown + 1}
                    </div>
                  )}
                  
                  {/* Current number (centered, large, bright) */}
                  <div
                    className="text-8xl font-bold text-white transition-all duration-500"
                    style={{
                      textShadow: "0 0 30px rgba(255,255,255,0.9), 0 0 60px rgba(255,255,255,0.5)",
                      transform: "scale(1)",
                    }}
                  >
                    {countdown}
                  </div>
                  
                  {/* Next number (appears to rotate from left) */}
                  {countdown > 1 && (
                    <div
                      className="absolute text-4xl font-bold text-white transition-all duration-1000"
                      style={{
                        transform: `translateX(-70px) translateY(-10px) scale(0.6) rotateY(-45deg)`,
                        opacity: 0.2,
                        filter: "blur(1px)",
                      }}
                    >
                      {countdown - 1}
                    </div>
                  )}
                </div>
              </div>
              
              {/* "Hold Still" text when face detected */}
              {isFaceDetected && (
                <div className="mt-2 text-center">
                  <p className="animate-pulse text-sm font-semibold uppercase tracking-[0.2em] text-green-400">
                    Hold Still
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Perfect placement message - shown below oval when face is perfectly centered */}
          {isPerfectlyPlaced && countdown === null && (
            <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 whitespace-nowrap">
              <p 
                className="text-xs font-bold uppercase tracking-[0.25em] text-green-400 sm:text-sm"
                style={{
                  textShadow: "0 0 20px rgba(74, 222, 128, 0.8), 0 0 40px rgba(74, 222, 128, 0.5)",
                  animation: "pulse 1.5s ease-in-out infinite",
                }}
              >
                Perfect, Hold Still!
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Timer control - Left Side */}
      <div className="absolute bottom-1/4 left-4 flex translate-y-1/2 flex-row items-center gap-2 sm:bottom-1/2 sm:left-8">
        {/* Timer button */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setShowTimerOptions(!showTimerOptions);
          }}
          disabled={countdown !== null}
          className="relative z-10 flex h-16 w-16 cursor-pointer items-center justify-center rounded-full bg-white/80 backdrop-blur-sm transition hover:bg-white disabled:opacity-50"
          title="Timer"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="28"
            height="28"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
        </button>

        {/* Timer options oval */}
        {showTimerOptions && (
          <div 
            className="absolute left-12 flex h-16 items-center rounded-r-full bg-white/60 pl-8 pr-4 backdrop-blur-sm"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex gap-3">
              {(["OFF", "3s", "10s"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => {
                    setTimerSetting(option);
                    setShowTimerOptions(false);
                  }}
                  className={`cursor-pointer rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-widest transition ${
                    timerSetting === option
                      ? "bg-[#1A1B1C] text-white"
                      : "bg-white/50 text-[#1A1B1C] hover:bg-white/80"
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Capture button - Right Side */}
      <div className="absolute bottom-1/4 right-4 flex translate-y-1/2 flex-col items-center gap-2 sm:bottom-1/2 sm:right-8">
        <button
          type="button"
          onClick={handleCapture}
          disabled={!stream || !!error || countdown !== null}
          className="flex h-20 w-20 cursor-pointer items-center justify-center rounded-full border-4 border-white bg-white/20 backdrop-blur-sm transition hover:bg-white/30 disabled:opacity-50"
          title="Take Picture"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="40"
            height="40"
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
            <circle cx="12" cy="13" r="4" />
          </svg>
        </button>
      </div>

      {/* Instructions - Bottom */}
      <div className="absolute bottom-16 left-1/2 -translate-x-1/2 text-center sm:bottom-4">
        <p className="text-sm text-white/80">
          For best results, make sure you have
        </p>
        <p className="mt-2 text-sm text-white">
          <span className="inline-block">• Neutral Expression</span>
          <span className="mx-2">•</span>
          <span className="inline-block">Frontal Pose</span>
          <span className="mx-2">•</span>
          <span className="inline-block">Adequate Lighting</span>
        </p>
      </div>

      {/* Error message */}
      {error && (
        <div className="absolute left-1/2 top-20 -translate-x-1/2">
          <p className="rounded-lg bg-red-600 px-4 py-2 text-xs text-white">
            {error}
          </p>
        </div>
      )}

      {/* FaceDetector loading/not available warning */}
      {faceDetectorLoading && stream && !error && (
        <div className="absolute left-1/2 top-20 -translate-x-1/2 max-w-sm">
          <p className="rounded-lg bg-blue-600 px-4 py-2 text-xs text-white text-center flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
            Loading face detection...
          </p>
        </div>
      )}
      {!faceDetectorReady && !faceDetectorLoading && stream && !error && (
        <div className="absolute left-1/2 top-20 -translate-x-1/2 max-w-sm">
          <p className="rounded-lg bg-yellow-600 px-4 py-2 text-xs text-white text-center">
            Face detection failed to load. Please refresh the page and try again.
          </p>
        </div>
      )}

      {/* Close button - can press ESC or click this */}
      <button
        type="button"
        onClick={onClose}
        className="hidden sm:block absolute right-8 bottom-8 cursor-pointer text-xs uppercase tracking-[0.2em] text-white/80 transition hover:text-white"
      >
        Close (ESC)
      </button>
    </div>
  );
}
