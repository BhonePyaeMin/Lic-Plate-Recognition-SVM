export default function EngineBadge({ engine, fallback }) {
  const label = engine === 'matlab' ? 'MATLAB' : fallback ? 'OpenCV (MATLAB failed)' : 'OpenCV'
  return <span className={`badge badge--${engine}`}>{label}</span>
}
