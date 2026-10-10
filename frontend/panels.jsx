// True when a keydown target is a form field, link, button, or contenteditable
// element — the global quiz/flashcard shortcuts must not hijack typing or
// activate controls (e.g. Space on a Save button would flip the card).
const isEditableTarget = (t) => !!(t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.tagName === 'BUTTON' || t.tagName === 'A' || t.isContentEditable));

const QuizPanel = ({ source, onBack, artifact, onElaborate, compact, reviewItems = [], onSaveReviewItem }) => {
  const questionsData = artifact?.content_json?.questions || artifact?.questions || [];
  const useReal = questionsData.length > 0;
  const quiz = useReal ? questionsData : [];
  const title = artifact?.title || 'Quiz';
  const artifactSourceDocIds = artifact?.source_document_ids || artifact?.sourceDocumentIds || [];
  const primaryArtifactSourceDocId = artifactSourceDocIds[0] || '';

  // ---- normalize helpers (same as before) ----
  const normalizeQuestion = React.useCallback((q, idx) => {
    if (useReal) {
      const choices = (q.choices || []).map((c, j) => {
        const label = String.fromCharCode(97 + j);
        return { id: label, text: String(c).replace(/^[A-D]\.\s*/, ''), label };
      });
      const answerText = String(q.answer || '').trim();
      const correctLabel = answerText.replace(/^[A-D][.)]\s*/, '');
      const matchedId = choices.find(c => {
        const raw = String(c.text).trim();
        return raw === correctLabel.trim()
            || raw === answerText
            || c.id === answerText.toLowerCase();
      })?.id;
      const correct = matchedId || answerText.toLowerCase();
      return {
        id: `q-${idx}`,
        reviewIndex: idx,
        type: 'mc',
        prompt: q.question,
        options: choices,
        correct: correct,
        rationale: q.explanation,
        source_document_id: q.source_document_id || primaryArtifactSourceDocId,
        source_chunk_id: q.source_chunk_id || '',
        section_path: q.section_path || '',
        page_start: Number.isFinite(Number(q.page_start)) ? Number(q.page_start) : -1,
        page_end: Number.isFinite(Number(q.page_end)) ? Number(q.page_end) : -1,
        citation: {
          src: q.section_path || source?.title || '',
          loc: q.page_start > 0 ? `p. ${q.page_start}` : '',
          section: q.section_path || '',
        },
      };
    }
    return {
      ...q,
      id: q.id || `q-${idx}`,
      reviewIndex: idx,
      source_document_id: q.source_document_id || primaryArtifactSourceDocId,
      source_chunk_id: q.source_chunk_id || '',
      section_path: q.section_path || q.citation?.section || '',
      page_start: Number.isFinite(Number(q.page_start)) ? Number(q.page_start) : -1,
      page_end: Number.isFinite(Number(q.page_end)) ? Number(q.page_end) : -1,
    };
  }, [useReal, source, primaryArtifactSourceDocId]);

  // Normalize all questions once
  const normalized = React.useMemo(() => {
    if (!Array.isArray(quiz) || quiz.length === 0) return [];
    return quiz.map((q, i) => normalizeQuestion(q, i));
  }, [quiz, normalizeQuestion]);
  const total = normalized.length;

  // ---- state ----
  const [order, setOrder] = React.useState(() => normalized.map((_, i) => i));
  const [currentIdx, setCurrentIdx] = React.useState(0);
  const [selectedAnswer, setSelectedAnswer] = React.useState(null);
  const [revealed, setRevealed] = React.useState(false);
  const [answers, setAnswers] = React.useState({}); // { qId: { picked: 'a', correct: bool } }
  const [finished, setFinished] = React.useState(false);
  const [isShuffled, setIsShuffled] = React.useState(false);

  // Reset order when quiz data changes
  React.useEffect(() => {
    setOrder(normalized.map((_, i) => i));
    setCurrentIdx(0);
    setSelectedAnswer(null);
    setRevealed(false);
    setAnswers({});
    setFinished(false);
    setIsShuffled(false);
  }, [normalized]);

  if (total === 0) {
    return (
      <div className="h-full overflow-y-auto scroll-thin">
        <div className="max-w-[760px] mx-auto px-8 py-8">
          {onBack && (
            <button onClick={onBack} className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted hover:text-ink inline-flex items-center gap-1.5 mb-4">
              <Icon name="chevronLeft" size={12} /> back to study guide
            </button>
          )}
          <div className="serif-display text-[20px] text-ink">No questions in this quiz.</div>
        </div>
      </div>
    );
  }

  const currentQuestion = normalized[order[currentIdx]];
  const answeredCount = Object.keys(answers).length;
  const correctCount = Object.values(answers).filter(a => a.correct).length;
  const hasAnsweredCurrent = !!answers[currentQuestion?.id];
  const currentAnswer = answers[currentQuestion?.id];
  const artifactId = artifact?.id || artifact?.artifact_id || '';
  const currentReviewIndex = currentQuestion?.reviewIndex ?? order[currentIdx] ?? currentIdx;
  const currentReviewItem = artifactId
    ? (reviewItems || []).find(i =>
        i.artifact_id === artifactId
        && i.item_type === 'quiz_question'
        && Number(i.item_index) === Number(currentReviewIndex)
      )
    : null;

  const formatOptionAnswer = (question, optionId) => {
    if (!question || !optionId) return '';
    const option = (question.options || []).find(o => o.id === optionId);
    if (!option) return String(optionId).toUpperCase();
    return `${String(option.id).toUpperCase()}. ${option.text}`;
  };

  const buildQuizReviewPayload = (question, answerPayload = {}, flags = {}) => {
    const picked = answerPayload.picked || selectedAnswer || '';
    const index = question?.reviewIndex ?? order[currentIdx] ?? currentIdx;
    return {
      artifact_id: artifactId,
      item_type: 'quiz_question',
      item_index: index,
      prompt: question?.prompt || '',
      answer: formatOptionAnswer(question, question?.correct),
      choices: (question?.options || []).map(o => `${String(o.id).toUpperCase()}. ${o.text}`),
      explanation: question?.rationale || '',
      source_document_id: question?.source_document_id || primaryArtifactSourceDocId,
      source_chunk_id: question?.source_chunk_id || '',
      section_path: question?.section_path || question?.citation?.section || '',
      page_start: Number.isFinite(Number(question?.page_start)) ? Number(question.page_start) : -1,
      page_end: Number.isFinite(Number(question?.page_end)) ? Number(question.page_end) : -1,
      is_starred: !!flags.is_starred,
      is_mistake: !!flags.is_mistake,
      last_user_answer: formatOptionAnswer(question, picked),
    };
  };

  const saveQuizReviewItem = (question, answerPayload = {}, flags = {}) => {
    if (!artifactId || !onSaveReviewItem || !question) return Promise.resolve(null);
    return onSaveReviewItem(buildQuizReviewPayload(question, answerPayload, flags));
  };

  // ---- actions ----
  const handleSelectOption = (optionId) => {
    if (revealed) return;
    setSelectedAnswer(optionId);
  };

  const handleCheckAnswer = () => {
    if (!selectedAnswer || !currentQuestion) return;
    const isCorrect = selectedAnswer === currentQuestion.correct;
    const answerPayload = { picked: selectedAnswer, correct: isCorrect };
    setRevealed(true);
    setAnswers(prev => ({ ...prev, [currentQuestion.id]: answerPayload }));
    if (!isCorrect) {
      saveQuizReviewItem(currentQuestion, answerPayload, { is_mistake: true })
        .catch(err => console.warn('Failed to save quiz mistake:', err));
    }
  };

  const handleNext = () => {
    if (currentIdx >= total - 1) {
      setFinished(true);
    } else {
      setCurrentIdx(i => i + 1);
      setSelectedAnswer(null);
      setRevealed(false);
    }
  };

  const handlePrev = () => {
    if (currentIdx > 0) {
      setCurrentIdx(i => i - 1);
      // Restore previous answer state
      const prevQ = normalized[order[currentIdx - 1]];
      const prevAnswer = answers[prevQ?.id];
      if (prevAnswer) {
        setSelectedAnswer(prevAnswer.picked);
        setRevealed(true);
      } else {
        setSelectedAnswer(null);
        setRevealed(false);
      }
    }
  };

  const handleReshuffle = () => {
    // Fisher-Yates shuffle of all indices
    const arr = [...order];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    setOrder(arr);
    setCurrentIdx(0);
    setSelectedAnswer(null);
    setRevealed(false);
    setAnswers({});
    setFinished(false);
    setIsShuffled(true);
  };

  const handleRestart = () => {
    // Keep current shuffle but reset all answers
    setCurrentIdx(0);
    setSelectedAnswer(null);
    setRevealed(false);
    setAnswers({});
    setFinished(false);
  };

  const handleStartFresh = () => {
    // Reset everything including shuffle
    setOrder(normalized.map((_, i) => i));
    setCurrentIdx(0);
    setSelectedAnswer(null);
    setRevealed(false);
    setAnswers({});
    setFinished(false);
    setIsShuffled(false);
  };

  const handleReview = () => {
    setCurrentIdx(0);
    setSelectedAnswer(null);
    setRevealed(false);
    setFinished(false);
  };

  // ---- keyboard ----
  React.useEffect(() => {
    const handler = (e) => {
      if (finished) return;
      if (isEditableTarget(e.target)) return;
      if (e.key === 'ArrowRight' && revealed) handleNext();
      else if (e.key === 'ArrowLeft') handlePrev();
      else if (e.key === 'Enter' && selectedAnswer && !revealed) handleCheckAnswer();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentIdx, selectedAnswer, revealed, finished, order]);

  // ---- summary screen ----
  if (finished) {
    const pct = total > 0 ? Math.round((correctCount / total) * 100) : 0;
    return (
      <div className="h-full overflow-y-auto scroll-thin">
        <div className="max-w-[760px] mx-auto px-8 py-8">
          <button onClick={onBack} className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted hover:text-ink inline-flex items-center gap-1.5 mb-4">
            <Icon name="chevronLeft" size={12} /> back to study guide
          </button>
          <div className="text-center mt-12">
            <div className="mono text-[10.5px] uppercase tracking-[0.2em] text-muted mb-2">Quiz Complete</div>
            <h1 className="serif-display text-[36px] leading-[1.05] text-ink mb-2">{title}</h1>
            <div className="serif-display" style={{ fontSize: 'calc(56px * var(--folio-font-scale, 1))', fontWeight: 400, color: 'var(--accent)' }}>
              {correctCount}/{total}
            </div>
            <div className="text-[15px] text-muted mt-1 mb-6">{pct}% correct</div>

            {/* Percentage bar */}
            <div className="mx-auto max-w-[360px] h-2 rounded-full bg-line2 overflow-hidden mb-8">
              <div className="h-full transition-all duration-700"
                   style={{ width: `${pct}%`, background: pct >= 70 ? 'var(--ok)' : pct >= 40 ? '#D4850A' : 'var(--warn)' }} />
            </div>

            <div className="flex items-center justify-center gap-3 flex-wrap">
              <button onClick={handleReshuffle}
                className="sans rounded-lg border px-4 py-2.5 text-[12px] transition-colors inline-flex items-center gap-2"
                style={{ borderColor: 'var(--border-soft)', background: 'var(--surface-card)', color: 'var(--text-primary)' }}>
                <Icon name="refresh" size={13} /> Reshuffle &amp; Retry
              </button>
              <button onClick={handleReview}
                className="sans rounded-lg border px-4 py-2.5 text-[12px] transition-colors"
                style={{ borderColor: 'var(--border-soft)', background: 'var(--surface-card)', color: 'var(--text-primary)' }}>
                Review Answers
              </button>
              <button onClick={handleStartFresh}
                className="sans rounded-lg px-4 py-2.5 text-[12px] transition-colors"
                style={{ color: 'var(--text-muted)' }}>
                Start Fresh
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ---- question view ----
  return (
    <div className={compact ? '' : 'h-full overflow-y-auto scroll-thin'}>
      <div className={compact ? '' : 'max-w-[760px] mx-auto px-8 py-8'}>
        {onBack && (
          <button onClick={onBack} className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted hover:text-ink inline-flex items-center gap-1.5 mb-4">
            <Icon name="chevronLeft" size={12} /> back to study guide
          </button>
        )}

        {/* Header row */}
        <div className="flex items-end justify-between gap-6 flex-wrap mb-4">
          <div>
            <div className="mono text-[10.5px] uppercase tracking-[0.2em] text-muted mb-2">Quiz - auto-generated</div>
            <h1 className="serif-display text-[28px] leading-[1.05] text-ink">{title}</h1>
          </div>
          <div className="text-[13px] text-muted">
            {answeredCount} of {total} answered - {correctCount} correct
          </div>
        </div>

        {/* Top bar: progress + actions */}
        <div className="mt-2 mb-3 flex items-center justify-between">
          <span className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted">
            question {currentIdx + 1} of {total}
            {isShuffled && <span className="ml-2" style={{ color: 'var(--accent)' }}>(shuffled)</span>}
          </span>
          <div className="flex items-center gap-3">
            <button onClick={handleReshuffle}
              className="sans text-[10.5px] uppercase tracking-wider hover:text-[var(--accent)] transition-colors"
              style={{ color: 'var(--text-muted)' }}>
              Shuffle
            </button>
            <button onClick={handleRestart}
              className="sans text-[10.5px] uppercase tracking-wider hover:text-[var(--accent)] transition-colors"
              style={{ color: 'var(--text-muted)' }}>
              Restart
            </button>
          </div>
        </div>

        {/* Progress dots */}
        <div className="flex items-center gap-1 mb-6 flex-wrap">
          {order.map((origIdx, i) => {
            const qId = normalized[origIdx]?.id;
            const ans = answers[qId];
            return (
              <span key={i} className="h-1 rounded-full transition-all"
                style={{
                  width: i === currentIdx ? 20 : 12,
                  background: i === currentIdx ? 'var(--accent)'
                    : ans ? (ans.correct ? 'var(--ok)' : 'var(--warn)') : 'var(--border-soft)',
                }} />
            );
          })}
        </div>

        {/* Question card */}
        <Card className="p-6 mb-5">
          <div className="flex items-baseline gap-3 mb-4">
            <span className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted">
              Q{String(currentIdx + 1).padStart(2, '0')} - multiple choice
            </span>
          </div>
          <div className="serif-display text-[20px] leading-snug text-ink mb-5">{currentQuestion.prompt}</div>

          {currentQuestion.options && (
            <div className="space-y-2.5">
              {currentQuestion.options.map(o => {
                const isSelected = selectedAnswer === o.id;
                const isCorrectOption = revealed && o.id === currentQuestion.correct;
                const isWrongPick = revealed && isSelected && o.id !== currentQuestion.correct;
                return (
                  <button
                    key={o.id}
                    onClick={() => handleSelectOption(o.id)}
                    disabled={revealed}
                    className={cx(
                      'w-full text-left rounded-lg border px-4 py-3 flex items-start gap-3 transition-all',
                      isCorrectOption ? 'border-ok bg-ok/10' :
                      isWrongPick ? 'border-warn bg-warn/10' :
                      isSelected ? 'border-ink bg-cream' : 'border-line bg-card hover:border-ink/40',
                      revealed && !isCorrectOption && !isWrongPick ? 'opacity-60' : ''
                    )}
                  >
                    <span className={cx('mt-0.5 w-6 h-6 rounded-full border-2 flex items-center justify-center mono text-[11px] shrink-0 font-medium',
                      isCorrectOption ? 'border-ok text-ok bg-ok/10' :
                      isWrongPick ? 'border-warn text-warn bg-warn/10' :
                      isSelected ? 'border-ink text-ink bg-ink/10' : 'border-line2 text-muted')}>
                      {o.id.toUpperCase()}
                    </span>
                    <span className="text-[14px] text-ink leading-snug pt-0.5">{o.text}</span>
                  </button>
                );
              })}
            </div>
          )}
        </Card>

        {/* Action bar */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2">
            {currentQuestion.citation && (
              <span className="cite-chip rounded-md text-[11.5px] mono px-2.5 py-1.5 text-ink2 inline-flex items-center gap-1.5"
                style={{ background: 'var(--surface-card)', border: '1px solid var(--border-soft)' }}>
                <Icon name="quote" size={11} className="text-indigo-500" />
                {currentQuestion.citation.src} {currentQuestion.citation.loc && `- ${currentQuestion.citation.loc}`}
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {revealed && (
              <>
                {onElaborate && (
                  <button
                    onClick={() => onElaborate(currentQuestion, currentAnswer)}
                    className="sans inline-flex items-center gap-1.5 rounded-full px-3 py-2 text-[11px] uppercase tracking-wider transition-all"
                    style={{
                      color: 'var(--surface-main)',
                      background: 'var(--accent)',
                      border: '1px solid var(--accent)',
                      boxShadow: 'var(--shadow-ring)',
                    }}
                    title="Explore this question in chat"
                  >
                    <Icon name="sparkle" size={13} /> Elaborate in chat
                  </button>
                )}
                {artifactId && onSaveReviewItem && (
                  <button
                    onClick={() => saveQuizReviewItem(currentQuestion, currentAnswer || { picked: selectedAnswer, correct: selectedAnswer === currentQuestion.correct }, { is_starred: true, is_mistake: currentAnswer?.correct === false })
                      .catch(err => console.warn('Failed to save quiz review item:', err))}
                    className="sans inline-flex items-center gap-1.5 rounded-full px-3 py-2 text-[11px] uppercase tracking-wider transition-all"
                    style={{
                      color: currentReviewItem ? 'var(--accent)' : 'var(--text-secondary)',
                      background: 'var(--surface-card)',
                      border: '1px solid var(--border-soft)',
                    }}
                    title={currentReviewItem ? (currentReviewItem.is_mistake ? 'Saved as mistake' : 'Saved') : 'Save'}
                  >
                    <Icon name="star" size={13} />
                    {currentReviewItem ? (currentReviewItem.is_mistake ? 'Saved as mistake' : 'Saved') : 'Save'}
                  </button>
                )}
                <span className={cx('mono text-[10.5px] uppercase tracking-[0.18em] font-normal',
                  currentAnswer?.correct ? 'text-ok' : 'text-warn')}>
                  {currentAnswer?.correct ? 'Correct' : 'Incorrect'}
                </span>
              </>
            )}

            {!revealed ? (
              <button
                onClick={handleCheckAnswer}
                disabled={!selectedAnswer}
                className="sans rounded-full px-4 py-2 text-[12px] transition-colors"
                style={{
                  background: selectedAnswer ? 'var(--accent)' : 'var(--border-soft)',
                  border: `1px solid ${selectedAnswer ? 'var(--accent)' : 'var(--border-soft)'}`,
                  color: selectedAnswer ? 'var(--surface-main)' : 'var(--text-muted)',
                  cursor: selectedAnswer ? 'pointer' : 'not-allowed',
                }}>
                Check Answer
              </button>
            ) : (
              <button onClick={handleNext}
                className="sans rounded-full px-4 py-2 text-[12px] transition-colors"
                style={{ background: 'var(--accent)', border: '1px solid var(--accent)', color: 'var(--surface-main)' }}>
                {currentIdx >= total - 1 ? 'Finish' : 'Next ->'}
              </button>
            )}
          </div>
        </div>

        {/* Explanation after reveal */}
        {revealed && (
          <div className="mt-4 rounded-lg bg-cream border border-line2 px-4 py-3.5 fade-in">
            <div className="mono text-[10px] uppercase tracking-[0.18em] text-muted mb-1.5">
              {currentAnswer?.correct ? 'Explanation' : `Correct answer: ${currentQuestion.correct.toUpperCase()}`}
            </div>
            <div className="text-[13.5px] text-ink2 leading-relaxed">{currentQuestion.rationale}</div>
          </div>
        )}

        {/* Keyboard hint */}
        <div className="text-center sans mt-6" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)' }}>
          {revealed ? '-> Next question' : '<- / -> Navigate'} - Enter to check answer
        </div>

        {/* Prev / Next nav */}
        <div className="flex justify-between mt-4 pb-8">
          <button onClick={handlePrev} disabled={currentIdx === 0}
            className="sans text-[11px] uppercase tracking-wider"
            style={{ color: currentIdx === 0 ? 'var(--border-medium)' : 'var(--accent)', cursor: currentIdx === 0 ? 'default' : 'pointer' }}>
            {'<- Previous'}
          </button>
          {revealed && (
            <button onClick={handleNext}
              className="sans text-[11px] uppercase tracking-wider"
              style={{ color: 'var(--accent)', cursor: 'pointer' }}>
              {currentIdx >= total - 1 ? 'Finish' : 'Next ->'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

const FlashcardsPanel = ({ source, onBack, artifact, reviewItems = [], onSaveReviewItem }) => {
  // Use real artifact data if available
  const cardsData = artifact?.content_json?.cards || artifact?.cards || window.FOLIO_DATA?.FLASHCARDS || [];
  const title = artifact?.title || 'Flashcards';
  return (
    <FlashcardsViewer cards={cardsData} title={title} onBack={onBack} artifact={artifact} reviewItems={reviewItems} onSaveReviewItem={onSaveReviewItem} />
  );
};

// Shared flashcards viewer ...used by both FlashcardsPanel and Studio detail
const FlashcardsViewer = ({ cards, title, onBack, compact, artifact, reviewItems = [], onSaveReviewItem }) => {
  const [idx, setIdx] = React.useState(0);
  const [flipped, setFlipped] = React.useState(false);
  const [order, setOrder] = React.useState(() => {
    // Defensive: ensure cards is an array before calling .map
    if (!Array.isArray(cards) || cards.length === 0) return [];
    return cards.map((_, i) => i);
  });
  const [isShuffled, setIsShuffled] = React.useState(false);

  const cardCount = Array.isArray(cards) ? cards.length : 0;
  const artifactId = artifact?.id || artifact?.artifact_id || '';

  React.useEffect(() => {
    if (!Array.isArray(cards) || cards.length === 0) {
      setOrder([]);
      setIdx(0);
      setFlipped(false);
      setIsShuffled(false);
      return;
    }
    setOrder(cards.map((_, i) => i));
    setIdx(0);
    setFlipped(false);
    setIsShuffled(false);
  }, [cardCount, artifactId]);

  if (cardCount === 0) {
    return (
      <div className={compact ? '' : 'h-full overflow-y-auto scroll-thin'}>
        <div className={compact ? '' : 'max-w-[760px] mx-auto px-8 py-8'}>
          <div className="serif-display text-[20px] text-ink">No cards in this deck.</div>
        </div>
      </div>
    );
  }

  const originalCardIndex = order[idx] ?? idx;
  const card = cards[originalCardIndex] || {};
  const savedCard = artifactId
    ? (reviewItems || []).find(i =>
        i.artifact_id === artifactId
        && i.item_type === 'flashcard'
        && Number(i.item_index) === Number(originalCardIndex)
      )
    : null;
  const next = () => { setIdx(i => Math.min(cardCount - 1, i + 1)); setFlipped(false); };
  const prev = () => { setIdx(i => Math.max(0, i - 1)); setFlipped(false); };
  const flip = () => setFlipped(f => !f);

  const handleShuffle = () => {
    const remaining = order.filter((_, i) => i !== idx);
    for (let i = remaining.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
    }
    setOrder([order[idx], ...remaining]);
    setIdx(0);
    setFlipped(false);
    setIsShuffled(true);
  };

  const handleRestart = () => {
    setOrder(cards.map((_, i) => i));
    setIdx(0);
    setFlipped(false);
    setIsShuffled(false);
  };

  const handleSaveCard = (event) => {
    event.stopPropagation();
    event.preventDefault();
    if (!artifactId || !onSaveReviewItem || !card) return;
    onSaveReviewItem({
      artifact_id: artifactId,
      item_type: 'flashcard',
      item_index: originalCardIndex,
      prompt: card.term || '',
      answer: card.definition || '',
      choices: [],
      explanation: '',
      source_document_id: card.source_document_id || artifact?.source_document_ids?.[0] || '',
      source_chunk_id: card.source_chunk_id || '',
      section_path: card.section_path || '',
      page_start: Number.isFinite(Number(card.page_start)) ? Number(card.page_start) : -1,
      page_end: Number.isFinite(Number(card.page_end)) ? Number(card.page_end) : -1,
      is_starred: true,
      is_mistake: false,
    }).catch(err => console.warn('Failed to save flashcard review item:', err));
  };

  React.useEffect(() => {
    const handler = (e) => {
      if (isEditableTarget(e.target)) return;
      if (e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); flip(); }
      else if (e.key === 'ArrowRight') next();
      else if (e.key === 'ArrowLeft') prev();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [idx, flipped]);

  return (
    <div className={compact ? '' : 'h-full overflow-y-auto scroll-thin'}>
      <div className={compact ? '' : 'max-w-[760px] mx-auto px-8 py-8'}>
        {onBack && <button onClick={onBack} className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted hover:text-ink inline-flex items-center gap-1.5 mb-4"><Icon name="chevronLeft" size={12} /> back to study guide</button>}
        {!compact && (
          <div className="flex items-end justify-between gap-6 flex-wrap">
            <div>
              <div className="mono text-[10.5px] uppercase tracking-[0.2em] text-muted mb-2">Flashcards</div>
              <h1 className="serif-display text-[36px] leading-[1.05] text-ink">{title}</h1>
              <div className="text-[13px] text-muted mt-1">{cardCount} cards in this deck</div>
            </div>
          </div>
        )}

        <div className="mt-6 mb-3 flex items-center justify-between">
          <span className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted">
            card {idx + 1} of {cardCount}
            {isShuffled && <span className="ml-2" style={{ color: 'var(--accent)' }}>(shuffled)</span>}
          </span>
            <button onClick={handleShuffle}
                    className="sans text-[10.5px] uppercase tracking-wider hover:text-[var(--accent)] transition-colors"
                    style={{ color: 'var(--text-muted)' }}>
              Shuffle
            </button>
            <button onClick={handleRestart}
                    className="sans text-[10.5px] uppercase tracking-wider hover:text-[var(--accent)] transition-colors"
                    style={{ color: 'var(--text-muted)' }}>
              Restart
            </button>
          </div>

        {/* Progress dots */}
        <div className="flex items-center gap-1 mb-5">
          {order.map((_, i) => (
            <span key={i} className="h-1 rounded-full transition-all"
                  style={{
                    width: i === idx ? 20 : 12,
                    background: i === idx ? 'var(--accent)'
                      : i < idx ? 'var(--ok)' : 'var(--border-soft)'
                  }} />
          ))}
        </div>

        {/* Card */}
        <div onClick={flip} className="cursor-pointer mb-5 mx-auto" style={{
          position: 'relative',
          minHeight: 280, maxWidth: 560, padding: '48px 32px',
          background: 'var(--surface-card)',
          borderRadius: 12,
          border: '1px solid var(--border-soft)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          transition: 'transform 150ms',
        }}>
          {artifactId && onSaveReviewItem && (
            <button
              type="button"
              onClick={handleSaveCard}
              className="absolute top-3 right-3 inline-flex items-center justify-center rounded-full transition-colors"
              style={{
                width: 34,
                height: 34,
                color: savedCard ? '#D4850A' : 'var(--text-muted)',
                background: savedCard ? 'rgba(212, 133, 10, 0.10)' : 'var(--surface-main)',
                border: '1px solid var(--border-soft)',
              }}
              title={savedCard ? 'Saved' : 'Save'}
              aria-label={savedCard ? 'Saved' : 'Save'}
            >
              <Icon name="star" size={15} />
            </button>
          )}
          {!flipped ? (
            <>
              <div className="mono text-[10px] uppercase tracking-[0.2em] mb-4" style={{ color: 'var(--text-muted)' }}>Term</div>
              <div className="serif-display text-center" style={{ fontSize: 'calc(28px * var(--folio-font-scale, 1))', fontWeight: 500, color: 'var(--text-primary)' }}>
                {card.term}
              </div>
            </>
          ) : (
            <>
              <div className="mono text-[10px] uppercase tracking-[0.2em] mb-4" style={{ color: 'var(--text-muted)' }}>Definition</div>
              <div className="serif-display text-center" style={{ fontSize: 'calc(18px * var(--folio-font-scale, 1))', lineHeight: 1.5, color: 'var(--text-primary)' }}>
                {card.definition}
              </div>
              {(card.section_path || card.page_start > 0) && (
                <div className="sans mt-4" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-muted)' }}>
                  {card.section_path}
                  {card.page_start > 0 && ` - p. ${card.page_start}`}
                </div>
              )}
            </>
          )}
        </div>

        <div className="text-center sans" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)' }}>
          Click or Space to {flipped ? 'flip back' : 'reveal'} - {'<- / ->'} to navigate
        </div>

        <div className="flex justify-between mt-6">
          <button onClick={prev} disabled={idx === 0}
            className="sans text-[11px] uppercase tracking-wider" style={{ color: idx === 0 ? 'var(--border-medium)' : 'var(--accent)', cursor: idx === 0 ? 'default' : 'pointer' }}>
            {'<- Previous'}
          </button>
          <button onClick={next} disabled={idx >= cardCount - 1}
            className="sans text-[11px] uppercase tracking-wider" style={{ color: idx >= cardCount - 1 ? 'var(--border-medium)' : 'var(--accent)', cursor: idx >= cardCount - 1 ? 'default' : 'pointer' }}>
            Next ->
          </button>
        </div>
      </div>
    </div>
  );
};

const SummaryPanel = ({ source, onBack, artifact }) => {
  // If an artifact is provided, render its markdown content
  if (artifact && artifact.markdown) {
    const hasMarked = typeof window !== 'undefined' && window.marked;
    const hasPurify = typeof window !== 'undefined' && window.DOMPurify;
    const stats = artifact.stats || {};

    return (
    <div className={compact ? '' : 'h-full overflow-y-auto scroll-thin'}>
      <div className={compact ? '' : 'max-w-[760px] mx-auto px-8 py-8'}>
        {onBack && (
          <button onClick={onBack} className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted hover:text-ink inline-flex items-center gap-1.5 mb-4">
            <Icon name="chevronLeft" size={12} /> back to study guide
          </button>
        )}
          <div className="mono text-[10.5px] uppercase tracking-[0.2em] text-muted mb-2">Summary</div>
          <h1 className="serif-display text-[36px] leading-[1.05] text-ink">{(window._cleanDisplayTitle && window._cleanDisplayTitle(artifact.title)) || artifact.title || 'Summary'}</h1>
          {(stats.chunk_count != null || stats.scope_layer) && (
            <div className="sans mt-2 flex flex-wrap gap-3" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
              {stats.chunk_count != null && <span>{stats.chunk_count} chunks</span>}
              {stats.scope_layer && <span>scope: {stats.scope_layer}</span>}
              {stats.page_start > 0 && <span>pp. {stats.page_start}-{stats.page_end}</span>}
            </div>
          )}
          <div className="dotted-line my-6" />

          <Card className="p-6">
            {hasMarked && hasPurify ? (
              <div
                className="markdown-body serif-display"
                style={{ fontSize: 'calc(17px * var(--folio-font-scale, 1))', lineHeight: 1.6, color: 'var(--text-primary)' }}
                dangerouslySetInnerHTML={{ __html: window.DOMPurify.sanitize(window.marked.parse(artifact.markdown, { gfm: true, breaks: false })) }}
              />
            ) : (
              <div className="serif-display whitespace-pre-wrap" style={{ fontSize: 'calc(17px * var(--folio-font-scale, 1))', lineHeight: 1.6, color: 'var(--text-primary)' }}>
                {artifact.markdown}
              </div>
            )}
          </Card>
        </div>
      </div>
    );
  }

  // Fallback: legacy hardcoded summary
  return null;
};

// -- Mind map helpers -------------------------------------------------------
const _mindmapSlug = (text, fallback) => {
  const s = String(text || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return s || fallback || ('node-' + Math.random().toString(36).slice(2, 8));
};

const _shortMindmapLabel = (text) => {
  let s = String(text || '').replace(/\[[^\]]+\]/g, '').replace(/[*_`#]/g, '').trim();
  s = s.replace(/^[-*\s]*/, '').replace(/\s+/g, ' ');
  if (!s) return 'Untitled';

  const split = s.split(/\s+-\s+|:\s+/)[0].trim();
  if (split && split.length >= 4) s = split;

  const words = s.split(/\s+/).filter(Boolean);
  if (words.length > 9) s = words.slice(0, 9).join(' ');
  if (s.length > 68) {
    const cut = s.slice(0, 68).trim();
    const last = cut.lastIndexOf(' ');
    s = last > 24 ? cut.slice(0, last) : cut;
  }
  const dangling = /(?:\b(?:a|an|the|and|or|of|to|for|from|with|without|by|in|on|into|that|which|as|is|are|can|could|would|should)\s*)$/i;
  while (dangling.test(s) && s.split(/\s+/).length > 2) {
    s = s.replace(dangling, '').trim();
  }
  return s || 'Untitled';
};

const _normalizeMindmapNode = (node, parentPath = [], fallbackSourceIds = [], indexPath = '0') => {
  if (!node || typeof node !== 'object') return null;
  const label = _shortMindmapLabel(node.label || node.title || node.prompt_term);
  const sourceIds = Array.isArray(node.source_document_ids) && node.source_document_ids.length
    ? node.source_document_ids
    : fallbackSourceIds;
  const normalized = {
    id: node.id || _mindmapSlug(label, `node-${indexPath}`),
    label,
    summary: String(node.summary || node.description || '').trim(),
    prompt_term: node.prompt_term || label,
    context_path: Array.isArray(node.context_path) ? node.context_path : parentPath,
    source_document_ids: sourceIds,
    children: [],
  };
  const children = Array.isArray(node.children) ? node.children : [];
  normalized.children = children
    .map((child, i) => _normalizeMindmapNode(child, [...parentPath, label], sourceIds, `${indexPath}-${i}`))
    .filter(Boolean);
  return normalized;
};

const parseMindmapMarkdownToTree = (markdown, title, sourceDocumentIds = []) => {
  const lines = String(markdown || '').split(/\r?\n/);
  const root = {
    id: 'root',
    label: _shortMindmapLabel(title || 'Mind Map'),
    prompt_term: _shortMindmapLabel(title || 'Mind Map'),
    context_path: [],
    source_document_ids: sourceDocumentIds,
    children: [],
  };
  const stack = [{ level: 1, node: root }];
  let counter = 0;

  lines.forEach(raw => {
    const line = raw.trim();
    if (!line) return;

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const label = _shortMindmapLabel(heading[2]);
      if (level === 1 && root.children.length === 0) {
        root.label = label;
        root.prompt_term = label;
        return;
      }
      while (stack.length > 1 && stack[stack.length - 1].level >= level) stack.pop();
      const parent = stack[stack.length - 1].node;
      const node = {
        id: _mindmapSlug(label, `node-${counter++}`),
        label,
        prompt_term: label,
        context_path: [...(parent.context_path || []), parent.label],
        source_document_ids: sourceDocumentIds,
        children: [],
      };
      parent.children.push(node);
      stack.push({ level, node });
      return;
    }

    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      const parent = stack[stack.length - 1].node;
      const label = _shortMindmapLabel(bullet[1]);
      parent.children.push({
        id: _mindmapSlug(label, `leaf-${counter++}`),
        label,
        prompt_term: label,
        context_path: [...(parent.context_path || []), parent.label],
        source_document_ids: sourceDocumentIds,
        children: [],
      });
    }
  });

  return root.children.length ? root : null;
};

const buildMindmapPrompt = (node, rootLabel, title) => {
  const term = node?.prompt_term || node?.label || 'this concept';
  const path = Array.isArray(node?.context_path) ? node.context_path.filter(Boolean) : [];
  const context = path.length ? path[path.length - 1] : (rootLabel || title || '');
  const childTerms = Array.isArray(node?.children)
    ? node.children.map(c => c?.prompt_term || c?.label).filter(Boolean).slice(0, 6)
    : [];
  const childHint = childTerms.length
    ? ` Cover these subtopics where the sources support them: ${childTerms.join(', ')}.`
    : '';
  const summaryHint = node?.summary ? ` The card summary is: ${node.summary}.` : '';
  if (context && context !== term) {
    return `Discuss what these sources say about ${term}, in the larger context of ${context}.${summaryHint}${childHint} Use specific evidence and cite the sources.`;
  }
  return `Discuss what these sources say about ${term}.${summaryHint}${childHint} Use specific evidence and cite the sources.`;
};

const MindmapTreeNode = ({ node, depth, expanded, onToggle, onAskNode }) => {
  const hasChildren = Array.isArray(node.children) && node.children.length > 0;
  const isOpen = expanded.has(node.id);
  const bg = depth === 0 ? '#575a76' : depth === 1 ? '#3f4852' : '#2f463f';
  const border = depth === 0 ? 'rgba(168,172,255,0.35)' : 'rgba(195,210,224,0.10)';

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onMouseDown={(e) => e.stopPropagation()}
        onClick={() => onAskNode && onAskNode(node)}
        className="sans text-left transition-transform hover:scale-[1.015] focus:outline-none focus:ring-2 focus:ring-white/30"
        title={node.summary || 'Ask chat about this concept'}
        style={{
          minWidth: depth === 0 ? 180 : 180,
          maxWidth: depth >= 2 ? 340 : 300,
          background: bg,
          border: `1px solid ${border}`,
          borderRadius: 7,
          color: 'var(--text-primary)',
          padding: depth === 0 ? '14px 18px' : '12px 16px',
          fontSize: depth === 0 ? 17 : 16,
          lineHeight: 1.2,
          overflowWrap: 'anywhere',
          boxShadow: '0 1px 0 rgba(255,255,255,0.04), 0 8px 20px rgba(0,0,0,0.18)',
        }}
      >
        <span style={{ display: 'block', fontWeight: 400 }}>{node.label}</span>
        {node.summary && (
          <span
            style={{
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
              marginTop: 6,
              color: 'rgba(255,255,255,0.72)',
              fontSize: depth === 0 ? 12 : 11.5,
              lineHeight: 1.25,
            }}
          >
            {node.summary}
          </span>
        )}
      </button>
      {hasChildren && (
        <button
          type="button"
          onMouseDown={(e) => e.stopPropagation()}
          onClick={(e) => { e.stopPropagation(); onToggle(node.id); }}
          className="sans shrink-0 flex items-center justify-center transition-colors"
          aria-label={isOpen ? 'Collapse branch' : 'Expand branch'}
          title={isOpen ? 'Collapse' : 'Expand'}
          style={{
            width: 24,
            height: 24,
            borderRadius: 999,
            background: '#46515d',
            color: 'var(--text-primary)',
            fontSize: 'calc(14px * var(--folio-font-scale, 1))',
            lineHeight: 1,
          }}
        >
          {isOpen ? '<' : '>'}
        </button>
      )}
    </div>
  );
};

const MindmapBranch = ({ node, depth, expanded, onToggle, onAskNode }) => {
  const open = expanded.has(node.id);
  const children = Array.isArray(node.children) ? node.children : [];
  return (
    <div className="flex items-center gap-6">
      <MindmapTreeNode node={node} depth={depth} expanded={expanded} onToggle={onToggle} onAskNode={onAskNode} />
      {open && children.length > 0 && (
        <div
          className="relative flex flex-col gap-5 py-2 pl-8"
          style={{ borderLeft: '2px solid rgba(178,190,255,0.55)' }}
        >
          {children.map((child) => (
            <div key={child.id} className="relative">
              <span
                aria-hidden
                style={{
                  position: 'absolute',
                  left: -32,
                  top: '50%',
                  width: 32,
                  borderTop: '2px solid rgba(178,190,255,0.55)',
                }}
              />
              <MindmapBranch node={child} depth={depth + 1} expanded={expanded} onToggle={onToggle} onAskNode={onAskNode} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const InteractiveMindmapViewer = ({ tree, title, compact, onBack, onAskNode }) => {
  const normalizedTree = React.useMemo(
    () => _normalizeMindmapNode(tree, [], tree?.source_document_ids || [], '0'),
    [tree]
  );
  const initialExpanded = React.useMemo(() => {
    const ids = new Set();
    if (normalizedTree) ids.add(normalizedTree.id);
    return ids;
  }, [normalizedTree]);
  const [expanded, setExpanded] = React.useState(initialExpanded);

  React.useEffect(() => {
    setExpanded(initialExpanded);
  }, [initialExpanded]);

  const viewportRef = React.useRef(null);
  const dragRef = React.useRef(null);
  const [view, setView] = React.useState({ x: compact ? 44 : 72, y: compact ? 120 : 150, scale: 1 });
  const [dragging, setDragging] = React.useState(false);

  React.useEffect(() => {
    setView({ x: compact ? 44 : 72, y: compact ? 120 : 150, scale: 1 });
  }, [compact, normalizedTree?.id]);

  React.useEffect(() => {
    const onMove = (e) => {
      const drag = dragRef.current;
      if (!drag) return;
      e.preventDefault();
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      const nextX = drag.x + dx;
      const nextY = drag.y + dy;
      setView(prev => ({ ...prev, x: nextX, y: nextY }));
    };
    const onUp = () => {
      dragRef.current = null;
      setDragging(false);
      document.body.style.userSelect = '';
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      document.body.style.userSelect = '';
    };
  }, []);

  if (!normalizedTree) {
    return <div className="serif-display text-[16px] text-muted italic">No mind map data available.</div>;
  }

  const toggle = (id) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const resetView = () => {
    setView({ x: compact ? 44 : 72, y: compact ? 120 : 150, scale: 1 });
  };

  const zoomBy = (factor) => {
    setView(prev => ({
      ...prev,
      scale: Math.max(0.45, Math.min(2.2, prev.scale * factor)),
    }));
  };

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startY: e.clientY, x: view.x, y: view.y };
    setDragging(true);
    document.body.style.userSelect = 'none';
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    setView(prev => {
      const nextScale = Math.max(0.45, Math.min(2.2, prev.scale * factor));
      const contentX = (mouseX - prev.x) / prev.scale;
      const contentY = (mouseY - prev.y) / prev.scale;
      return {
        scale: nextScale,
        x: mouseX - contentX * nextScale,
        y: mouseY - contentY * nextScale,
      };
    });
  };

  return (
    <div className={compact ? '' : 'h-full overflow-auto'}>
      <div className={compact ? '' : 'max-w-full mx-auto px-8 py-8'}>
        {onBack && (
          <button onClick={onBack} className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted hover:text-ink inline-flex items-center gap-1.5 mb-4">
            <Icon name="chevronLeft" size={12} /> back
          </button>
        )}
        {!compact && (
          <div className="mb-5">
            <div className="mono text-[10.5px] uppercase tracking-[0.2em] text-muted mb-2">Mind Map</div>
            <h1 className="serif-display text-[36px] leading-[1.05] text-ink">{title || normalizedTree.label}</h1>
          </div>
        )}
        <div
          ref={viewportRef}
          className="scroll-thin"
          onMouseDown={handleMouseDown}
          onWheel={handleWheel}
          style={{
            position: 'relative',
            overflow: 'hidden',
            minHeight: compact ? 520 : 650,
            background: '#20252b',
            borderRadius: 8,
            border: '1px solid rgba(255,255,255,0.08)',
            padding: compact ? 34 : 48,
            cursor: dragging ? 'grabbing' : 'grab',
          }}
        >
          <div
            className="sans absolute right-3 top-3 z-10 flex items-center gap-1.5"
            onMouseDown={(e) => e.stopPropagation()}
            style={{ color: '#dfe7ef', fontSize: 'calc(11px * var(--folio-font-scale, 1))' }}
          >
            <button type="button" onClick={() => zoomBy(0.9)} title="Zoom out"
                    style={{ width: 28, height: 26, borderRadius: 6, background: 'rgba(255,255,255,0.08)' }}>-</button>
            <button type="button" onClick={resetView} title="Reset view"
                    style={{ height: 26, padding: '0 9px', borderRadius: 6, background: 'rgba(255,255,255,0.08)' }}>
              {Math.round(view.scale * 100)}%
            </button>
            <button type="button" onClick={() => zoomBy(1.1)} title="Zoom in"
                    style={{ width: 28, height: 26, borderRadius: 6, background: 'rgba(255,255,255,0.08)' }}>+</button>
          </div>
          <div style={{
            minWidth: 900,
            display: 'inline-flex',
            alignItems: 'center',
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            transformOrigin: '0 0',
            transition: dragging ? 'none' : 'transform 80ms ease-out',
          }}>
            <MindmapBranch node={normalizedTree} depth={0} expanded={expanded} onToggle={toggle} onAskNode={onAskNode} />
          </div>
        </div>
        <div className="sans mt-3" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)' }}>
          Click a label to ask the main chat. Drag the canvas to pan; scroll to zoom.
        </div>
      </div>
    </div>
  );
};

// Plain-text fallback when a mind-map artifact has no parseable tree.
const MindmapTextFallback = ({ markdown, title, onBack, compact }) => {
  const safeMarkdown = String(markdown || '');
  const html = (window.marked && window.DOMPurify)
    ? window.DOMPurify.sanitize(window.marked.parse(safeMarkdown, { gfm: true, breaks: false }))
    : null;
  return (
    <div className={compact ? '' : 'h-full overflow-auto'}>
      <div className={compact ? '' : 'max-w-full mx-auto px-8 py-8'}>
        {onBack && <button onClick={onBack} className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted hover:text-ink inline-flex items-center gap-1.5 mb-4"><Icon name="chevronLeft" size={12} /> back</button>}
        <div className="mono text-[10.5px] uppercase tracking-[0.2em] text-muted mb-2">Mind Map (text fallback)</div>
        <h1 className="serif-display text-[28px] leading-[1.05] text-ink mb-4">{title || 'Mind Map'}</h1>
        {html
          ? <div className="markdown-body serif-display" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.75, color: 'var(--text-primary)' }} dangerouslySetInnerHTML={{ __html: html }} />
          : <pre className="serif-display whitespace-pre-wrap" style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', lineHeight: 1.6, color: 'var(--text-primary)' }}>{safeMarkdown}</pre>}
      </div>
    </div>
  );
};

const MindmapViewer = ({ tree, markdown, title, onBack, compact, onAskNode, sourceDocumentIds }) => {
  const parsedTree = React.useMemo(() => {
    if (tree && tree.children) return tree;
    return parseMindmapMarkdownToTree(markdown, title, sourceDocumentIds || []);
  }, [tree, markdown, title, sourceDocumentIds]);

  if (parsedTree && parsedTree.children && parsedTree.children.length > 0) {
    return (
      <InteractiveMindmapViewer
        tree={parsedTree}
        title={title}
        compact={compact}
        onBack={onBack}
        onAskNode={onAskNode}
      />
    );
  }

  return <MindmapTextFallback markdown={markdown} title={title} onBack={onBack} compact={compact} />;
};

const MindmapPanel = ({ source, onBack, artifact }) => {
  const tree = artifact?.content_json?.tree || artifact?.tree || null;
  const markdown = artifact?.content_json?.markdown || artifact?.markdown || '';
  const title = artifact?.title || (source ? `Mind Map: ${source.title}` : 'Mind Map');
  return (tree || markdown) ? (
    <MindmapViewer
      tree={tree}
      markdown={markdown}
      title={title}
      onBack={onBack}
      sourceDocumentIds={artifact?.source_document_ids || []}
    />
  ) : (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="max-w-[760px] mx-auto px-8 py-8">
        {onBack && <button onClick={onBack} className="mono text-[10.5px] uppercase tracking-[0.18em] text-muted hover:text-ink inline-flex items-center gap-1.5 mb-4"><Icon name="chevronLeft" size={12} /> back</button>}
        <div className="serif-display text-[20px] text-ink">No mind map data available.</div>
      </div>
    </div>
  );
};

Object.assign(window, {
  QuizPanel,
  QuizViewer: QuizPanel,
  FlashcardsPanel,
  FlashcardsViewer,
  SummaryPanel,
  MindmapViewer,
  MindmapPanel,
  InteractiveMindmapViewer,
  parseMindmapMarkdownToTree,
  buildMindmapPrompt,
});
