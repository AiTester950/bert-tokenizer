package com.example.java_bert.tokenization;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.*;

public class TokenizerUtils {

    public static String cleanText(String text) {
        // Performs invalid character removal and whitespace cleanup on text."""
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            Character c = text.charAt(i);
            int cp = (int) c;
            if (cp == 0 || cp == 0xFFFD || isControl(c)) {
                continue;
            }
            if (isWhitespace(c)) {
                output.append(" ");
            } else {
                output.append(c);
            }
        }
        return output.toString();
    }

    public static String tokenizeChineseChars(String text) {
        // Adds whitespace around any CJK character.
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            Character c = text.charAt(i);
            int cp = (int) c;
            if (isChineseChar(cp)) {
                output.append(" ");
                output.append(c);
                output.append(" ");
            } else {
                output.append(c);
            }
        }
        return output.toString();
    }

    public static List<String> whitespaceTokenize(String text) {
        // Runs basic whitespace cleaning and splitting on a piece of text.
        text = text.trim();
        if (text.length() > 0) {
            return Arrays.asList(text.split("\\s+"));
        }
        return new ArrayList<>();
    }

    public static String runStripAccents(String token) {
        token = Normalizer.normalize(token, Form.NFD);
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < token.length(); i++) {
            Character c = token.charAt(i);
            if (Character.NON_SPACING_MARK != Character.getType(c)) {
                output.append(c);
            }
        }
        return output.toString();
    }

    public static List<String> runSplitOnPunc(String token, List<String> neverSplit) {
        List<String> output = new ArrayList<>();
        if (neverSplit != null && neverSplit.contains(token)) {
            output.add(token);
            return output;
        }

        StringBuilder str = new StringBuilder();
        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            if (isPunctuation(c)) {
                if (str.length() > 0) {
                    output.add(str.toString());
                    str.setLength(0);
                }
                output.add(Character.toString(c));
            } else {
                if (str.length() > 0 && isPunctuation(str.charAt(str.length() - 1))) {
                    output.add(str.toString());
                    str.setLength(0);
                }
                str.append(c);
            }
        }
        if (str.length() > 0) {
            output.add(str.toString());
        }
        return output;
    }


    public static Map<String, Integer> generateTokenIdMap(InputStream file) throws IOException {
        HashMap<String, Integer> tokenIdMap = new HashMap<String, Integer>();
        if (file == null) {
            return tokenIdMap;
        }

        try (BufferedReader br = new BufferedReader(new InputStreamReader(file))) {

            String line;
            int index = 0;
            while ((line = br.readLine()) != null) {
                tokenIdMap.put(line, index);
                index += 1;
            }
        }
        return tokenIdMap;
    }

    private static boolean isPunctuation(char c) {
        int cp = (int) c;
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
            return true;
        }
        int charType = Character.getType(c);
        if (Character.CONNECTOR_PUNCTUATION == charType || Character.DASH_PUNCTUATION == charType
                || Character.END_PUNCTUATION == charType || Character.FINAL_QUOTE_PUNCTUATION == charType
                || Character.INITIAL_QUOTE_PUNCTUATION == charType || Character.OTHER_PUNCTUATION == charType
                || Character.START_PUNCTUATION == charType) {
            return true;
        }
        return false;
    }

    private static boolean isWhitespace(char c) {

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            return true;
        }

        int charType = Character.getType(c);
        if (Character.SPACE_SEPARATOR == charType) {
            return true;
        }
        return false;
    }

    private static boolean isControl(char c) {
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }

        int charType = Character.getType(c);
        if (Character.CONTROL == charType || Character.DIRECTIONALITY_COMMON_NUMBER_SEPARATOR == charType
                || Character.FORMAT == charType || Character.PRIVATE_USE == charType || Character.SURROGATE == charType
                || Character.UNASSIGNED == charType) {
            return true;
        }
        return false;
    }

    private static boolean isChineseChar(int cp) {

        if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF)
                || (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F)
                || (cp >= 0x2B820 && cp <= 0x2CEAF) || (cp >= 0xF900 && cp <= 0xFAFF)
                || (cp >= 0x2F800 && cp <= 0x2FA1F)) {
            return true;
        }

        return false;
    }


}