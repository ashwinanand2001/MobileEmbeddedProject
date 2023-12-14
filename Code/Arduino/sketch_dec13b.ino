#include <Arduino_LSM9DS1.h>

const int MAX_SAMPLES = 1000; // Adjust the size based on your needs
float storedSignature[MAX_SAMPLES][9]; // 3 values for acceleration, 3 for gyroscope, 3 for magnetometer
int signatureIndex = 0;

#define DURATION 5000 // Signature duration in milliseconds
#define THRESHOLD 0.1 // Adjust the threshold based on your requirements

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize LSM9DS1 sensor!");
    while (1);
  }

  Serial.println("Place the pen on a flat surface for calibration.");
  delay(2000);

  // Calibrate sensors if needed
  //IMU.calibrate();
  Serial.println("Calibration complete.");
}

void loop() {
  Serial.println("Collecting signature data...");
  delay(1000);

  // Collect signature data
  unsigned long startTime = millis();
  while (millis() - startTime < DURATION) {
    float accelX, accelY, accelZ, gyroX, gyroY, gyroZ, magX, magY, magZ;

    IMU.readAcceleration(accelX, accelY, accelZ);
    IMU.readGyroscope(gyroX, gyroY, gyroZ);
    IMU.readMagneticField(magX, magY, magZ);

    // Process and store the sensor data
    processAndStoreData(accelX,accelY,accelZ,gyroX, gyroY, gyroZ,magX, magY, magZ);
  }

  // Authenticate signature using a threshold-based comparison
  if (authenticateSignature()) {
    Serial.println("Signature authenticated successfully!");
  } else {
    Serial.println("Authentication failed. Signature does not match.");
  }

  delay(5000); // Delay between signature attempts
}

void processAndStoreData(float accelX, float accelY, float accelZ, float gyroX, float gyroY, float gyroZ, float magX, float magY, float magZ) {
  // Store the sensor data in the array
  storedSignature[signatureIndex][0] = accelX;
  storedSignature[signatureIndex][1] = accelY;
  storedSignature[signatureIndex][2] = accelZ;
  storedSignature[signatureIndex][3] = gyroX;
  storedSignature[signatureIndex][4] = gyroY;
  storedSignature[signatureIndex][5] = gyroZ;
  storedSignature[signatureIndex][6] = magX;
  storedSignature[signatureIndex][7] = magY;
  storedSignature[signatureIndex][8] = magZ;

  // Increment the index for the next data point
  signatureIndex++;

  // Check if the array is full
  if (signatureIndex >= MAX_SAMPLES) {
    // Perform any additional processing or storage (e.g., save to file or database)
    saveSignatureToFile();

    // Reset the index for the next signature
    signatureIndex = 0;
  }
}

void saveSignatureToFile() {
  // Placeholder: In a real application, you would save the signature data to a file or database
  // Here, we print the data to the Serial monitor as an example

  Serial.println("Saving signature data...");

  for (int i = 0; i < MAX_SAMPLES; i++) {
    for (int j = 0; j < 9; j++) {
      Serial.print(storedSignature[i][j]);
      Serial.print(" ");
    }
    Serial.println(); // Move to the next line for the next set of data
  }

  Serial.println("Signature data saved.");
}

bool authenticateSignature() {
  // Placeholder for the stored signature profile (adjust these values based on your calibration)
  float storedAccelX = 0.0;
  float storedAccelY = 0.0;
  float storedAccelZ = 9.8;  // Assuming 1g for calibration
  float storedGyroX = 0.0;
  float storedGyroY = 0.0;
  float storedGyroZ = 0.0;
  float storedMagX = 0.0;
  float storedMagY = 0.0;
  float storedMagZ = 0.0;

  // Placeholder for live signature data (replace these values with actual live data)
  float liveAccelX = 0.0;
  float liveAccelY = 0.0;
  float liveAccelZ = 9.8;
  float liveGyroX = 0.0;
  float liveGyroY = 0.0;
  float liveGyroZ = 0.0;
  float liveMagX = 0.0;
  float liveMagY = 0.0;
  float liveMagZ = 0.0;

  // Calculate the differences or distances between live and stored data
  float accelDiff = abs(liveAccelX - storedAccelX) + abs(liveAccelY - storedAccelY) + abs(liveAccelZ - storedAccelZ);
  float gyroDiff = abs(liveGyroX - storedGyroX) + abs(liveGyroY - storedGyroY) + abs(liveGyroZ - storedGyroZ);
  float magDiff = abs(liveMagX - storedMagX) + abs(liveMagY - storedMagY) + abs(liveMagZ - storedMagZ);

  // Calculate a total difference or distance metric
  float totalDiff = accelDiff + gyroDiff + magDiff;

  // Adjust the threshold based on your requirements
  float threshold = THRESHOLD;

  // Compare the total difference with the threshold
  return (totalDiff <= threshold);
} 


/*#include <Arduino_LSM9DS1.h>




void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");


  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }


  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.print(" Hz, Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.print(" Hz, Magnetometer sample rate = ");
  Serial.print(IMU.magneticFieldSampleRate());
  Serial.println(" Hz");
  Serial.println("X_acc\tY_acc\tZ_acc\tX_gyro\tY_gyro\tZ_gyro\tX_mag\tY_mag\tZ_mag");


}


void loop() {
  float x_acc, y_acc, z_acc;
  float x_gyro, y_gyro, z_gyro;
  float x_mag, y_mag, z_mag;


  // Read accelerometer
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x_acc, y_acc, z_acc);
  }


  // Read gyroscope
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(x_gyro, y_gyro, z_gyro);
  }


  // Read magnetometer
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(x_mag, y_mag, z_mag);
  }


  // Print values
  Serial.print(x_acc);
  Serial.print('\t');
  Serial.print(y_acc);
  Serial.print('\t');
  Serial.print(z_acc);
  Serial.print('\t');
  Serial.print(x_gyro);
  Serial.print('\t');
  Serial.print(y_gyro);
  Serial.print('\t');
  Serial.print(z_gyro);
  Serial.print('\t');
  Serial.print(x_mag);
  Serial.print('\t');
  Serial.print(y_mag);
  Serial.print('\t');
  Serial.println(z_mag);


  // Delay to achieve a 20Hz sample rate (50ms delay)
  delay(50);


}*/
