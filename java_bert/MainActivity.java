package com.example.java_bert;

import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import com.example.java_bert.tokenization.BertTokenizer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            testTokenizer();
        } catch (IOException | OrtException e) {
            Log.e(TAG, "Error in tokenizing or running ONNX model", e);
        }
    }

    private void testTokenizer() throws IOException, OrtException {
        Log.d(TAG, "Starting testTokenizer...");

        // Load and initialize the tokenizer
        File vocabFile = createTempFileFromAsset("vocab.txt");
        Log.d(TAG, "Vocab file created at: " + vocabFile.getAbsolutePath());
        BertTokenizer bertTokenizer = new BertTokenizer(vocabFile.getAbsolutePath());

        // Tokenize sample text
        String text = "Mister Aziz, Layards Broadway, Colombo 14";
        Log.d(TAG, "Text to tokenize: " + text);
        List<String> tokens = bertTokenizer.tokenize(text);
        Log.d(TAG, "Tokens: " + tokens.toString());

        // Convert tokens to IDs
        long[][] tokenIds = bertTokenizer.convertTokensToIds(tokens);
        Log.d(TAG, "Token IDs: " + Arrays.deepToString(tokenIds));

        // Create attention mask and token type IDs
        long[][] attentionMask = new long[tokenIds.length][tokenIds[0].length];
        long[][] tokenTypeIds = new long[tokenIds.length][tokenIds[0].length];
        for (int i = 0; i < tokenIds.length; i++) {
            Arrays.fill(attentionMask[i], 1);
            Arrays.fill(tokenTypeIds[i], 0);
        }
        Log.d(TAG, "Attention Mask: " + Arrays.deepToString(attentionMask));
        Log.d(TAG, "Token Type IDs: " + Arrays.deepToString(tokenTypeIds));

        // Load the ONNX model
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        File modelFile = createTempFileFromAsset("bert_ner_model.onnx");
        Log.d(TAG, "Model file created at: " + modelFile.getAbsolutePath());
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelFile.getAbsolutePath(), sessionOptions);
        Log.d(TAG, "ONNX model session created.");

        // Create tensors for the model input
        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, tokenIds);
        OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMask);
        OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(env, tokenTypeIds);

        Log.d(TAG, "Input IDs Tensor: " + Arrays.deepToString((long[][]) inputIdsTensor.getValue()));
        Log.d(TAG, "Attention Mask Tensor: " + Arrays.deepToString((long[][]) attentionMaskTensor.getValue()));
        Log.d(TAG, "Token Type IDs Tensor: " + Arrays.deepToString((long[][]) tokenTypeIdsTensor.getValue()));

        // Run the model
        Map<String, OnnxTensor> inputMap = Map.of(
                "input_ids", inputIdsTensor,
                "attention_mask", attentionMaskTensor,
                "token_type_ids", tokenTypeIdsTensor
        );

        Log.d(TAG, "Running the ONNX model...");
        try (OrtSession.Result results = session.run(inputMap)) {
            float[][][] logits = (float[][][]) results.get(0).getValue();
            Log.d(TAG, "Logits: " + Arrays.deepToString(logits));

            // Apply softmax to logits
            float[][][] probabilities = applySoftmax(logits);
            Log.d(TAG, "Probabilities after softmax: " + Arrays.deepToString(probabilities));

            // Get predicted label IDs
            long[][] predictedLabelIds = new long[probabilities.length][probabilities[0].length];
            for (int i = 0; i < probabilities.length; i++) {
                for (int j = 0; j < probabilities[i].length; j++) {
                    predictedLabelIds[i][j] = argmax(probabilities[i][j]);
                }
            }
            Log.d(TAG, "Predicted Label IDs: " + Arrays.deepToString(predictedLabelIds));

            // Map label IDs to labels
            String[] id2label = {"O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"};
            String[] predictedLabels = new String[predictedLabelIds.length];
            for (int i = 0; i < predictedLabelIds.length; i++) {
                predictedLabels[i] = id2label[(int) predictedLabelIds[0][i]];
            }
            Log.d(TAG, "Predicted Labels: " + Arrays.toString(predictedLabels));

            // Log the tokens and their predicted labels
            for (int i = 0; i < tokens.size(); i++) {
                Log.d(TAG, "Token: " + tokens.get(i) + ", Predicted Label: " + predictedLabels[i]);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error running the ONNX model: " + e.getMessage(), e);
        } finally {
            session.close();
            Log.d(TAG, "ONNX model session closed.");
        }

        Log.d(TAG, "testTokenizer completed.");
    }


    private float[][][] applySoftmax(float[][][] logits) {
        int batchSize = logits.length;
        int seqLength = logits[0].length;
        int numLabels = logits[0][0].length;
        float[][][] softmax = new float[batchSize][seqLength][numLabels];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                float[] logitsRow = logits[i][j];
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (float logit : logitsRow) {
                    if (logit > maxLogit) {
                        maxLogit = logit;
                    }
                }
                float sum = 0.0f;
                for (int k = 0; k < numLabels; k++) {
                    softmax[i][j][k] = (float) Math.exp(logitsRow[k] - maxLogit);
                    sum += softmax[i][j][k];
                }
                for (int k = 0; k < numLabels; k++) {
                    softmax[i][j][k] /= sum;
                }
            }
        }
        return softmax;
    }

    private long argmax(float[] array) {
        int idx = 0;
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                idx = i;
            }
        }
        return idx;
    }

    private File createTempFileFromAsset(String assetName) throws IOException {
        InputStream inputStream = getAssets().open(assetName);
        File tempFile = File.createTempFile(assetName, null, getCacheDir());
        tempFile.deleteOnExit();
        try (FileOutputStream out = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
        return tempFile;
    }
}
