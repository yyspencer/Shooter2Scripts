#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <set>
#include <utility>

using namespace std;
namespace fs = filesystem;

// --------- Helpers ----------
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

bool isCSVFile(const fs::path& filePath) {
    return filePath.extension() == ".csv";
}

vector<vector<string>> loadCSV(const string& filePath) {
    vector<vector<string>> data;
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filePath << endl;
        return {};
    }
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> row;
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }
    file.close();
    return data;
}

pair<int, int> findPupilColumns(const vector<string>& headerRow) {
    int leftPupilCol = -1, rightPupilCol = -1;
    for (size_t i = 0; i < headerRow.size(); i++) {
        string trimmedCell = trim(headerRow[i]);
        if (trimmedCell.find("leftPupil") != string::npos) {
            leftPupilCol = i;
        }
        if (trimmedCell.find("rightPupil") != string::npos) {
            rightPupilCol = i;
        }
    }
    return {leftPupilCol, rightPupilCol};
}

pair<int, int> findEventRows(const vector<vector<string>>& data, int eventColumn) {
    int rowFor02 = -1, rowForShook = -1;
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i].size() <= eventColumn) continue;
        string eventColumnValue = data[i][eventColumn];
        if (eventColumnValue.find("0.2 seconds") != string::npos && rowFor02 == -1) {
            rowFor02 = i;
        }
        if (eventColumnValue.find("shook") != string::npos && rowForShook == -1) {
            rowForShook = i;
        }
        if (rowFor02 != -1 && rowForShook != -1) break;
    }
    return {rowFor02, rowForShook};
}

// --------- Main ----------
int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);

    fs::path shookFolder = "shook";
    fs::path baselineFolder = shookFolder / "baseline";
    fs::path pupilFolder = "pupil size";

    vector<fs::path> folders = {shookFolder, baselineFolder};

    if (!fs::exists(pupilFolder)) {
        fs::create_directory(pupilFolder);
    }

    for (const auto& folder : folders) {
        if (!fs::exists(folder) || !fs::is_directory(folder)) continue;

        for (const auto& entry : fs::directory_iterator(folder)) {
            if (!fs::is_regular_file(entry.path()) || !isCSVFile(entry.path())) continue;

            string fileName = entry.path().filename().string();
            string fileIndex = fileName.substr(0, 5);
            cout << "Extracting pupil size data for file " << fileIndex << " in folder " << folder << endl;

            vector<vector<string>> data = loadCSV(entry.path().string());
            if (data.empty()) {
                cerr << " -> ERROR: Could not load CSV ❌\n";
                continue;
            }

            pair<int, int> pupilColumns = findPupilColumns(data[0]);
            if (pupilColumns.first == -1 || pupilColumns.second == -1) {
                cerr << " -> ERROR: 'leftPupil' or 'rightPupil' column not found ❌\n";
                continue;
            }

            int eventColumn = -1;
            for (size_t i = 0; i < data[0].size(); i++) {
                if (trim(data[0][i]).find("robotEvent") != string::npos) {
                    eventColumn = i;
                    break;
                }
            }
            if (eventColumn == -1) {
                cerr << " -> ERROR: 'robotEvent' column not found ❌\n";
                continue;
            }

            pair<int, int> eventRows = findEventRows(data, eventColumn);
            if (eventRows.first == -1 || eventRows.second == -1) {
                cerr << " -> ERROR: '0.2 seconds' or 'shook' event not found ❌\n";
                continue;
            }

            double beforeTime = 0.0, afterTime = 0.0;
            try {
                beforeTime = stod(data[eventRows.first][0]);
                afterTime = stod(data[eventRows.second][0]);
            } catch (...) {
                cerr << " -> ERROR: Invalid time value ❌\n";
                continue;
            }

            vector<pair<double, double>> pupilBefore, pupilAfter;

            for (size_t i = 1; i < data.size(); i++) {
                if (data[i].size() <= (unsigned)max(pupilColumns.first, pupilColumns.second))
                    continue;
                double timeValue, leftPupil, rightPupil;
                try {
                    timeValue = stod(data[i][0]);
                    leftPupil = stod(data[i][pupilColumns.first]);
                    rightPupil = stod(data[i][pupilColumns.second]);
                } catch (...) {
                    continue;
                }
                if (timeValue >= (beforeTime - 5.0) && timeValue <= beforeTime) {
                    pupilBefore.push_back({leftPupil, rightPupil});
                }
                if (timeValue >= afterTime && timeValue <= (afterTime + 5.0)) {
                    pupilAfter.push_back({leftPupil, rightPupil});
                }
            }

            string outFileName = (pupilFolder / (fileIndex + "pupil.txt")).string();
            ofstream outFile(outFileName);
            if (!outFile) {
                cerr << "Error: Could not write to file " << outFileName << '\n';
                continue;
            }

            for (const auto& p : pupilBefore) {
                outFile << p.first << " " << p.second << "\n";
            }
            outFile << "\n";
            for (const auto& p : pupilAfter) {
                outFile << p.first << " " << p.second << "\n";
            }
            outFile.close();
            cout << "Finished file " << fileIndex << "\n";
        }
    }

    cout << "Pupil size extraction complete.\n";
    return 0;
}