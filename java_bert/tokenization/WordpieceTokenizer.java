package com.example.java_bert.tokenization;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class WordpieceTokenizer implements Tokenizer  {
    private Map<String, Integer> vocab;
    private String unkToken;
    private int maxInputCharsPerWord;

    public WordpieceTokenizer(Map<String, Integer> vocab, String unkToken, int maxInputCharsPerWord) {
        this.vocab = vocab;
        this.unkToken = unkToken;
        this.maxInputCharsPerWord = maxInputCharsPerWord;
    }

    public WordpieceTokenizer(Map<String, Integer> vocab, String unkToken) {
        this.vocab = vocab;
        this.unkToken = unkToken;
        this.maxInputCharsPerWord = 100;
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> outputTokens = new ArrayList<>();
        for (String token : TokenizerUtils.whitespaceTokenize(text)) {
            if (token.length() > maxInputCharsPerWord) {
                outputTokens.add(unkToken);
                continue;
            }
            boolean isBad = false;
            int start = 0;
            List<String> subTokens = new ArrayList<>();
            while (start < token.length()) {
                int end = token.length();
                String curSubstr = "";
                while (start < end) {
                    String substr = token.substring(start, end);
                    if (start > 0) {
                        substr = "##" + substr;
                    }
                    if (vocab.containsKey(substr)) {
                        curSubstr = substr;
                        break;
                    }
                    end -= 1;
                }
                if (curSubstr.equals("")) {
                    isBad = true;
                    break;
                }
                subTokens.add(curSubstr);
                start = end;
            }
            if (isBad) {
                outputTokens.add(unkToken);
            } else {
                outputTokens.addAll(subTokens);
            }
        }
        return outputTokens;
    }

}