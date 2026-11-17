/** Decorative placeholder block. Parent regions announce loading via aria-busy. */
export default function Skeleton({ width, height, aspect, className = '', style }) {
  return (
    <span
      className={`skeleton ${className}`}
      aria-hidden="true"
      style={{ width, height, aspectRatio: aspect, ...style }}
    />
  )
}
