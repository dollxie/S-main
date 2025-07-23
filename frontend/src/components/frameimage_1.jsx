import React, { useEffect, useState } from "react";

function FrameImage({ frame, frameIndex, segmentIndex }) {
  const [imgSrc, setImgSrc] = useState(null);

  useEffect(() => {
    async function fetchImage() {
      try {
        const response = await fetch(`${API_URL}/inferred_frames/${frame.saved_frame_path}`, {
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
          mode: "cors",
          credentials: "omit",
        });
        if (!response.ok) throw new Error("Image fetch failed");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setImgSrc(url);
      } catch (err) {
        console.error("Error loading image:", err);
        setImgSrc(null); // fallback or show placeholder
      }
    }
    fetchImage();

    // Cleanup blob URL on unmount or path change
    return () => {
      if (imgSrc) {
        URL.revokeObjectURL(imgSrc);
      }
    };
  }, [frame.saved_frame_path]);

  return (
    <img
      key={frameIndex}
      src={imgSrc}
      alt={`Representative frame ${frameIndex + 1} for segment ${segmentIndex + 1}`}
      crossOrigin="anonymous"
      style={{ maxHeight: "400px", maxWidth: "400px", border: "1px solid #ddd", borderRadius: "4px" }}
    />
  );
}
