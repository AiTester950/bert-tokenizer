package com.example.java_bert.tokenization;

import java.util.List;

public interface Tokenizer {

    List<String> tokenize(String text);
}