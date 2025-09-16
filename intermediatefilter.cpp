#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <cctype>
#include <set>
#include <vector>
#include <iomanip>

namespace fs = std::filesystem;

// 取檔名前 5 個字元為索引
std::string extractIndex(const std::string& name) {
    return name.substr(0, 5);
}

// 轉小寫
std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

int main() {
    const std::string targetFolder = "intermediate";
    std::set<std::string> allIndices;             // 全部 index
    std::set<std::string> indicesWithIntermediate;// 有 intermediate 檔的 index
    std::vector<std::pair<fs::path, fs::path>> moves; // 待搬移清單

    // 一次走訪目前目錄
    for (const auto& entry : fs::directory_iterator(fs::current_path())) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".csv")   continue;

        std::string fileName = entry.path().filename().string();
        std::string idx      = extractIndex(fileName);
        allIndices.insert(idx);

        std::string lower = toLower(fileName);
        bool isIntermediate =  lower.find("intermediate") != std::string::npos &&
                              lower.find("tablet")       == std::string::npos;

        if (isIntermediate) {
            indicesWithIntermediate.insert(idx);
            fs::path dest = fs::path(targetFolder) / entry.path().filename();
            moves.emplace_back(entry.path(), dest);
        }
    }

    // 建立資料夾（若尚未存在）
    if (!fs::exists(targetFolder))
        fs::create_directory(targetFolder);

    // 執行搬移
    for (const auto& [src, dst] : moves) {
        try {
            fs::rename(src, dst);
            std::cout << "Moved: " << src << " -> " << dst << '\n';
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error moving " << src << ": " << e.what() << '\n';
        }
    }

    // 統計與輸出
    std::size_t total    = allIndices.size();
    std::size_t covered  = indicesWithIntermediate.size();
    std::size_t missing  = total - covered;
    double      percent  = total ? static_cast<double>(covered) / total * 100.0 : 0.0;

    std::cout << "\n Intermediate file coverage\n";
    std::cout << "Total: " << total   << '\n';
    std::cout << "Covered: " << covered << '\n';
    std::cout << "Missing: " << missing << '\n';
    std::cout << std::fixed << std::setprecision(2)
              << "Coverage" << percent << "%\n";

    if (missing > 0) {
        std::cout << "\nMissing: \n";
        for (const auto& idx : allIndices)
            if (!indicesWithIntermediate.count(idx))
                std::cout << idx << ' ';
        std::cout << '\n';
    }
    return 0;
}