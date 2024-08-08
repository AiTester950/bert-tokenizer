package com.example.java_bert.tokenization;

import java.util.ArrayList;
import java.util.List;

public class BasicTokenizer implements Tokenizer {
    private boolean doLowerCase;
    private List<String> neverSplit;
    private boolean tokenizeChineseChars;

    public BasicTokenizer(boolean doLowerCase, List<String> neverSplit, boolean tokenizeChineseChars) {
        this.doLowerCase = doLowerCase;
        if (neverSplit == null) {
            neverSplit = new ArrayList<>();
        }
        this.neverSplit = neverSplit;
        this.tokenizeChineseChars = tokenizeChineseChars;
    }

    public BasicTokenizer() {
        this(false, null, true);  // Default constructor should use doLowerCase = false
    }

    @Override
    public List<String> tokenize(String text) {
        text = TokenizerUtils.cleanText(text);
        if (tokenizeChineseChars) {
            text = TokenizerUtils.tokenizeChineseChars(text);
        }
        List<String> originalTokens = TokenizerUtils.whitespaceTokenize(text);

        List<String> splitTokens = new ArrayList<>();
        for (String token : originalTokens) {
            String processedToken = token;
            if (doLowerCase && !neverSplit.contains(token)) {
                processedToken = TokenizerUtils.runStripAccents(token.toLowerCase());
            }
            splitTokens.addAll(TokenizerUtils.runSplitOnPunc(processedToken, neverSplit));
        }
        return splitTokens;
    }
}
