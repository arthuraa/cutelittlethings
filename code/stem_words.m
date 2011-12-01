function [stem_idx, stem_counts] = stem_words(word_idx, word_counts, ...
                                              stems)
% STEM_WORDS -
%

[stem_idx, m, n] = unique(stems(word_idx));
stem_counts = zeros(numel(stem_idx), 1);
for i = 1:numel(word_idx)
    stem_counts(n(i)) = stem_counts(n(i)) + word_counts(i);
end

