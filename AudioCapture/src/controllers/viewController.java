package controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Pane;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;

import javax.sound.sampled.*;
import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Properties;

public class viewController {
    @FXML
    private ImageView buttonImageView;

    @FXML
    private Slider slider;

    @FXML
    private TextField directoryTextField;

    @FXML
    private Pane mainPane;

    @FXML
    private Label statusLabel;

    @FXML
    private Button directoryButton;

    private File wavFile;
    private AudioFileFormat.Type fileType = AudioFileFormat.Type.WAVE;
    private TargetDataLine dataLine;

    private Properties properties = new Properties();

    private AudioFormat getAudioFormat() {
        float sampleRate = 16000;
        int sampleSizeInBits = 8;
        int channels = 2;
        boolean signed = true;
        boolean bigEndian = true;
        return new AudioFormat(sampleRate, sampleSizeInBits, channels, signed, bigEndian);
    }

    @FXML
    private void initialize() {
        try {
            Path path = Paths.get(System.getProperty("user.home"), "AppData", "Local", "AudioCapture", "audiocapture.properties");
            File f = path.toFile();
            if(!f.exists()) {
                f.getParentFile().mkdir();
                properties.setProperty("defaultPath", System.getProperty("user.home"));
                properties.store(new FileOutputStream(path.toString()), "store to properties file");
            }
            properties.load(new FileInputStream(path.toString()));
            directoryTextField.setText(properties.getProperty("defaultPath"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @FXML
    private void clickButton() {
        Image image = buttonImageView.getImage();
        if (image.impl_getUrl().contains("play")) {
            image = new Image(getClass().getResource("../res/stop.png").toExternalForm());
            Path path = Paths.get(directoryTextField.getText());
            if (slider.getValue() == 1)
                path = Paths.get(directoryTextField.getText(), "correct");
            else if (slider.getValue() == 0)
                path = Paths.get(directoryTextField.getText(), "incorrect");

            File f = path.toFile();
            if (!f.isDirectory())
                f.mkdir();
            wavFile = Paths.get(path.toAbsolutePath().toString(), LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd-MM-yy_HH-mm-ss")) + ".wav").toFile();
            Thread t = new Thread(new StartRecording());
            t.start();
            statusLabel.setText("Recording " + wavFile.getName() + "...");
            slider.setDisable(true);
            directoryTextField.setDisable(true);
            directoryButton.setDisable(true);
        } else {
            image = new Image(getClass().getResource("../res/play.png").toExternalForm());
            Thread t2 = new Thread(new StopRecording());
            t2.start();
            statusLabel.setText(wavFile.getName() + " saved.");
            slider.setDisable(false);
            directoryTextField.setDisable(false);
            directoryButton.setDisable(false);
        }
        buttonImageView.setImage(image);
    }

    @FXML
    public void chooseDirectory() {
        DirectoryChooser directoryChooser = new DirectoryChooser();
        directoryChooser.setInitialDirectory(new File(properties.getProperty("defaultPath")));
        Stage stage = (Stage) mainPane.getScene().getWindow();
        File selectedDirectory = directoryChooser.showDialog(stage);

        if (selectedDirectory != null) {
            directoryTextField.setText(selectedDirectory.getAbsolutePath());
            properties.setProperty("defaultPath", selectedDirectory.getAbsolutePath());
            try {
                Path path = Paths.get(System.getProperty("user.home"), "AppData", "Local", "AudioCapture", "audiocapture.properties");
                properties.store(new FileOutputStream(path.toString()), "store to properties file");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    class StartRecording implements Runnable {
        @Override
        public void run() {
            try {
                AudioFormat audioFormat = getAudioFormat();
                DataLine.Info info = new DataLine.Info(TargetDataLine.class, audioFormat);

                //checks if system supports the data line
                if (!AudioSystem.isLineSupported(info)) {
                    System.err.println("DUPA, NIE MA WSPARCIA LINII PRZESYLOWEJ");
                    System.exit(0);
                }
                dataLine = (TargetDataLine) AudioSystem.getLine(info);
                dataLine.open(audioFormat);
                dataLine.start();
                AudioInputStream ais = new AudioInputStream(dataLine);
                AudioSystem.write(ais, fileType, wavFile);
            } catch (LineUnavailableException | IOException e) {
                e.printStackTrace();
            }
        }
    }

    class StopRecording implements Runnable {
        @Override
        public void run() {
            dataLine.stop();
            dataLine.close();
        }
    }
}
