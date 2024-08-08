package com.example.java_bert.tokenization;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import android.util.Log;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BertTokenizer implements Tokenizer {

    private String vocabFile = "vocab.txt";
    private Map<String, Integer> tokenIdMap;
    private Map<Integer, String> idTokenMap;
    private final boolean doLowerCase = false; // Updated to false
    private final boolean doBasicTokenize = true;
    private final List<String> neverSplit = new ArrayList<>();
    private final String unkToken = "[UNK]";
    private final String sepToken = "[SEP]";
    private final String padToken = "[PAD]";
    private final String clsToken = "[CLS]";
    private final String maskToken = "[MASK]";
    private final boolean tokenizeChineseChars = true;
    private BasicTokenizer basicTokenizer;
    private WordpieceTokenizer wordpieceTokenizer;

    private static final int MAX_LEN = 512;

    public BertTokenizer(String vocabFile) {
        this.vocabFile = vocabFile;
        init();
    }

    public BertTokenizer() {
        init();
    }

    private void init() {
        try {
            this.tokenIdMap = loadVocab(vocabFile);
        } catch (IOException e) {
            Log.e("BertTokenizer", "Unable to load vocab due to: ", e);
        }
        this.idTokenMap = new HashMap<>(this.tokenIdMap.size());
        for (String key : tokenIdMap.keySet()) {
            this.idTokenMap.put(tokenIdMap.get(key), key);
        }

        if (doBasicTokenize) {
            this.basicTokenizer = new BasicTokenizer(doLowerCase, neverSplit, tokenizeChineseChars);
        }
        this.wordpieceTokenizer = new WordpieceTokenizer(tokenIdMap, unkToken);
    }

    private Map<String, Integer> loadVocab(String vocabFileName) throws IOException {
        return TokenizerUtils.generateTokenIdMap(new FileInputStream(vocabFileName));
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> splitTokens = new ArrayList<>();
        if (doBasicTokenize) {
            for (String token : basicTokenizer.tokenize(text)) {
                splitTokens.addAll(wordpieceTokenizer.tokenize(token));
            }
        } else {
            splitTokens.addAll(wordpieceTokenizer.tokenize(text));
        }
        return splitTokens;
    }


    public String convertTokensToString(List<String> tokens) {
        StringBuilder sb = new StringBuilder();
        for (String token : tokens) {
            sb.append(token.replace("##", "")).append(" ");
        }
        return sb.toString().trim();
    }

    public long[][] convertTokensToIds(List<String> tokens) {
        long[][] result = new long[1][tokens.size() + 2];
        int i = 1;
        result[0][0] = tokenIdMap.get(clsToken);
        for (String s : tokens) {
            result[0][i++] = tokenIdMap.get(s);
        }
        result[0][i] = tokenIdMap.get(sepToken);
        return result;
    }

    static long[] paddingZero(long[] array1, int paddingSize) {
        long[] result = new long[array1.length + paddingSize];
        System.arraycopy(array1, 0, result, 0, array1.length);
        for (int i = array1.length; i < result.length; i++) {
            result[i] = 0;
        }
        return result;
    }

    public Map<String, OnnxTensor> tokenizeOnnxTensor(List<String> texts) throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        long[][] textTokensIds = new long[texts.size()][];
        long[][] typeIds = new long[texts.size()][];
        long[][] attentionMasks = new long[texts.size()][];

        int rowIndex = 0;
        int maxColumn = 0;
        for (String text : texts) {
            List<String> tokens = tokenize(text);
            long[] tokenIds = new long[tokens.size() + 2];
            int index = 0;
            tokenIds[index++] = tokenIdMap.get(clsToken);
            for (String s : tokens) {
                tokenIds[index++] = tokenIdMap.get(s);
            }
            tokenIds[index++] = tokenIdMap.get(sepToken);
            textTokensIds[rowIndex] = tokenIds;
            typeIds[rowIndex] = buildTokenTypeArray(index);
            attentionMasks[rowIndex] = buildAttentionMaskArray(index);
            maxColumn = Math.max(maxColumn, index);
            rowIndex++;
        }

        // Padding 0
        for (int row = 0; row < texts.size(); row++) {
            if (textTokensIds[row].length < maxColumn) {
                textTokensIds[row] = paddingZero(textTokensIds[row], maxColumn - textTokensIds[row].length);
                typeIds[row] = paddingZero(typeIds[row], maxColumn - typeIds[row].length);
                attentionMasks[row] = paddingZero(attentionMasks[row], maxColumn - attentionMasks[row].length);
            }
        }

        OnnxTensor inputIds = OnnxTensor.createTensor(env, textTokensIds);
        OnnxTensor tokenTypeIds = OnnxTensor.createTensor(env, typeIds);
        OnnxTensor attentionMask = OnnxTensor.createTensor(env, attentionMasks);

        Map<String, OnnxTensor> inputMap = new HashMap<>();
        inputMap.put("input_ids", inputIds);
        inputMap.put("token_type_ids", tokenTypeIds);
        inputMap.put("attention_mask", attentionMask);

        return inputMap;
    }

    private long[] buildAttentionMaskArray(int size) {
        long[] attentionMask = new long[size];
        for (int i = 0; i < size; i++) {
            attentionMask[i] = 1;
        }
        return attentionMask;
    }

    long[] buildTokenTypeArray(int size) {
        long[] typeIds = new long[size];
        for (int i = 0; i < size; i++) {
            typeIds[i] = 0;
        }
        return typeIds;
    }

    public int vocabSize() {
        return tokenIdMap.size();
    }
}
