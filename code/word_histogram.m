function histogram = word_histogram(data, n_words, n_bigrams)

total.body = zeros(n_words, 1);
entries.body = zeros(n_words, 1);
total.title = zeros(n_words, 1);
entries.title = zeros(n_words, 1);
if isfield(data, 'bigram_count)
    use_bigrams = 1;
    total.bigrams = zeros(n_bigrams, 1);
    entries.bigrams = zeros(n_bigrams, 1);
else
    use_bigrams = 0;
end

for i = 1:size(data, 2)
    total.body(data(i).word_idx) = ...
        total.body(data(i).word_idx) + ...
        double(data(i).word_count);
    entries.body(data(i).word_idx) = ...
        entries.body(data(i).word_idx) + 1;
    total.title(data(i).title_idx) = ...
        total..title(data(i).title_idx) + ...
        double(data(i).title_count);
    entries.body(data(i).title_idx) = ...
        entries.body(data(i).title_idx) + 1;

    if use_bigrams
        total.title(data(i).bigram_idx) = ...
            total..title(data(i).bigram_idx) + ...
            double(data(i).bigram_count);
        entries.body(data(i).bigram_idx) = ...
            entries.body(data(i).bigram_idx) + 1;
    end
end