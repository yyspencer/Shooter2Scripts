#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <cmath> // For variance calculation
#include <algorithm> // For trimming whitespace

using namespace std;
namespace fs = filesystem;

// Function to trim whitespace from a string
string trim(const string& str) {
    string trimmed = str;
    trimmed.erase(trimmed.begin(), find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
        return !isspace(ch);
    }));
    trimmed.erase(find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
        return !isspace(ch);
    }).base(), trimmed.end());
    return trimmed;
}

// Function to check if a file has a .csv extension
bool isCSVFile(const fs::path& filePath) {
    return filePath.extension() == ".csv";
}

// Function to extract the first 5 characters as an index
string extractIndex(const string& fileName) {
    return fileName.substr(0, 5); // First 5 characters represent the index
}

// Function to find the column that contains "robotEvent" as a substring
int findRobotEventColumn(const string& filePath) {
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filePath << endl;
        return -1; // Return -1 if file cannot be opened
    }

    string headerLine;
    if (!getline(file, headerLine)) {
        cerr << "Error: Empty file " << filePath << endl;
        return -1;
    }

    stringstream ss(headerLine);
    string cell;
    int columnIndex = 0;

    while (getline(ss, cell, ',')) {
        string trimmedCell = trim(cell); // Trim leading/trailing spaces

        // Check if "robotEvent" exists as a substring
        if (trimmedCell.find("robotEvent") != string::npos) {
            file.close();
            return columnIndex; // Found the correct column
        }
        columnIndex++;
    }

    file.close();
    return -1; // "robotEvent" column not found
}

// Function to extract time values for "0.2 seconds" and "shook" from a CSV file
pair<double, double> extractTimeValues(const string& filePath, int eventColumn) {
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filePath << endl;
        return {-1, -1}; // Return invalid values
    }

    string line;
    double timeFor02 = -1, timeForShook = -1; // Default to -1 (not found)
    bool firstRowSkipped = false; // To ensure we skip the header row

    while (getline(file, line)) {
        if (!firstRowSkipped) { // Skip header row
            firstRowSkipped = true;
            continue;
        }

        stringstream ss(line);
        vector<string> row;
        string cell;
        
        // Read all columns
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        // Ensure we have enough columns
        if (row.size() <= eventColumn) continue;

        // First column is the time value
        double timeValue;
        try {
            timeValue = stod(row[0]); // Convert first column to double
        } catch (...) {
            continue; // Skip this line and move to the next line
        }

        // The "robotEvent" column contains the keywords
        string eventColumnValue = row[eventColumn];

        // Check for "0.2 seconds" as a substring
        if (eventColumnValue.find("0.2 seconds") != string::npos && timeFor02 == -1) {
            timeFor02 = timeValue;
        }

        // Check for "shook" as a substring
        if (eventColumnValue.find("shook") != string::npos && timeForShook == -1) {
            timeForShook = timeValue;
        }

        // If both values are found, break early
        if (timeFor02 != -1 && timeForShook != -1){
            break;
        }
    }

    file.close();

    // Ensure "0.2 seconds" occurs before "shook"
    if (timeFor02 != -1 && timeForShook != -1 && timeFor02 > timeForShook) {
        return {-1, -1}; // Error: "0.2 seconds" should be before "shook"
    }

    return {timeFor02, timeForShook};
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);

    string path = "."; // Default to current directory
    fs::path shookFolder = fs::path(path) / "shook";

    cout << fixed << setprecision(3);
    cout << "Scanning CSV files in the shook folder..." << endl;

    // Check if the shook folder exists
    if (!fs::exists(shookFolder) || !fs::is_directory(shookFolder)) {
        cerr << "Error: 'shook' folder does not exist!" << endl;
        return 1;
    }

    vector<double> timeDifferences; // Stores all valid (shook - 0.2 seconds) values

    cout << "\n==== Time Extraction Report ====\n";

    for (const auto& entry : fs::directory_iterator(shookFolder)) {
        if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
            string fileName = entry.path().filename().string();
            string fileIndex = extractIndex(fileName);

            // Find the "robotEvent" column index
            int eventColumn = findRobotEventColumn(entry.path().string());
            if (eventColumn == -1) {
                cout << "Index " << fileIndex << " -> ERROR: 'robotEvent' column not found ❌" << endl;
                continue;
            }

            // Extract time values for "0.2 seconds" and "shook"
            pair<double, double> times = extractTimeValues(entry.path().string(), eventColumn);

            // Print result
            if (times.first == -1 || times.second == -1) {
                cout << "Index " << fileIndex << " -> ERROR ❌" << endl;
            } else {
                double timeDiff = times.second - times.first;
                timeDifferences.push_back(timeDiff);

                cout << "Index " << fileIndex << ", \"0.2 seconds\": " << times.first 
                     << ", \"shook\": " << times.second 
                     << ", Time Difference: " << timeDiff << endl;
            }
        }
    }

    // Calculate mean and variance
    if (!timeDifferences.empty()) {
        double sum = 0.0, mean, variance = 0.0;
        int count = timeDifferences.size();

        // Calculate mean
        for (double val : timeDifferences) {
            sum += val;
        }
        mean = sum / count;

        // Calculate variance
        for (double val : timeDifferences) {
            variance += (val - mean) * (val - mean);
        }
        variance /= count; // Population variance
        cout<<"count: "<<count<<'\n';
        // Print mean and variance
        cout << "\n==== Statistical Analysis ====\n";
        cout << "Mean Time Difference: " << mean << endl;
        cout << "Variance of Time Difference: " << variance << endl;
    } 
    else {
        cout << "\nNo valid time differences found. Unable to calculate mean and variance.\n";
    }

    return 0;
}