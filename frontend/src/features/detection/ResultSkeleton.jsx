import Skeleton from '../../components/Skeleton.jsx'

/** Mirrors ResultView block for block so the swap does not shift the layout. */
export default function ResultSkeleton({ dimensions, showProcessed = false }) {
  const aspect = dimensions ? `${dimensions.width} / ${dimensions.height}` : '4 / 3'
  return (
    <div className="result" data-testid="result-skeleton">
      <p className="status-label" role="status">
        <span className="status-label__pulse" aria-hidden="true" />
        Analyzing image
      </p>
      <div className="card plate-card">
        <Skeleton width="7rem" height="0.875rem" />
        <Skeleton width="min(18rem, 80%)" height="3rem" style={{ marginTop: 12 }} />
        <div className="plate-card__meta">
          <Skeleton width="6rem" height="1.5rem" />
          <Skeleton width="5rem" height="1.5rem" />
        </div>
      </div>
      <div className="figures">
        <div className="card figure">
          <Skeleton width="6rem" height="0.875rem" />
          <Skeleton width="100%" aspect={aspect} style={{ marginTop: 12 }} />
        </div>
        <div className="card figure">
          <Skeleton width="8rem" height="0.875rem" />
          <Skeleton width="100%" aspect={aspect} style={{ marginTop: 12 }} />
        </div>
      </div>
      {showProcessed && (
        <div className="card figure">
          <Skeleton width="9rem" height="0.875rem" />
          <Skeleton width="100%" aspect={aspect} style={{ marginTop: 12 }} />
        </div>
      )}
      <div className="card figure figure--crop">
        <Skeleton width="7rem" height="0.875rem" />
        <Skeleton width="min(22rem, 100%)" aspect="4 / 1" style={{ marginTop: 12 }} />
      </div>
      <div className="card details">
        <Skeleton width="5rem" height="0.875rem" />
        {[0, 1, 2, 3].map((i) => (
          <div className="details__row" key={i}>
            <Skeleton width="7rem" height="0.875rem" />
            <Skeleton width="9rem" height="0.875rem" />
          </div>
        ))}
      </div>
    </div>
  )
}
