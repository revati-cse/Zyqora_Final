import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import type { Group } from "three";

// Small low-poly two-compartment bin (Organic / Recyclable). No post-processing, no particles.
function Bin({ spin }: { spin: boolean }) {
  const group = useRef<Group>(null);
  useFrame((_, dt) => {
    if (spin && group.current) group.current.rotation.y += dt * 0.25;
  });
  return (
    <group ref={group} rotation={[0.18, -0.6, 0]}>
      {[-0.62, 0.62].map((x, i) => (
        <group key={x} position={[x, 0, 0]}>
          <mesh position={[0, -0.1, 0]}>
            <cylinderGeometry args={[0.52, 0.42, 1.3, 7]} />
            <meshStandardMaterial color={i === 0 ? "#7a9a63" : "#244b37"} flatShading roughness={0.85} />
          </mesh>
          <mesh position={[0, 0.62, 0]} rotation={[0, 0, i === 0 ? 0.12 : -0.12]}>
            <cylinderGeometry args={[0.58, 0.58, 0.12, 7]} />
            <meshStandardMaterial color={i === 0 ? "#8fb07a" : "#2f5e45"} flatShading roughness={0.8} />
          </mesh>
        </group>
      ))}
      <mesh position={[0, -0.82, 0]}>
        <boxGeometry args={[2.6, 0.08, 1.2]} />
        <meshStandardMaterial color="#c9cfc6" flatShading />
      </mesh>
    </group>
  );
}

export default function BinScene({ reducedMotion }: { reducedMotion: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const el = host.current;
    if (!el || !("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  }, []);
  // Render only while visible; with reduced motion render one static frame.
  return (
    <div ref={host} className="h-full w-full" aria-hidden>
      <Canvas frameloop={reducedMotion ? "demand" : visible ? "always" : "never"} dpr={[1, 1.5]} camera={{ position: [0, 0.4, 4.2], fov: 38 }}
        gl={{ antialias: true, powerPreference: "low-power" }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[3, 4, 2]} intensity={1.1} />
        <Bin spin={!reducedMotion} />
      </Canvas>
    </div>
  );
}
