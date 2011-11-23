function histogram = word_histogram(data, word, title, bigram, ...
                                    n_words, n_bigrams)
% WORD_HISTOGRAM -
%
if word
    histogram.body = zeros(n_words, 1);
end
if title
    histogram.title = zeros(n_words, 1);
end
if bigram
    histogram.bigram = zeros(n_bigrams, 1);
end

for i = 1:size(data, 2)
    if word
        histogram.body(data(i).word_idx) = ...
            histogram.body(data(i).word_idx) + ...
            double(data(i).word_count);
    end
    if title
        histogram.title(data(i).title_idx) = ...
            histogram.title(data(i).title_idx) + ...
            double(data(i).title_count);
    end
    if bigram
        histogram.bigram(data(i).bigram_idx) = ...
            histogram.bigram(data(i).bigram_idx) + ...
            double(data(i).bigram_count);
    end
end
