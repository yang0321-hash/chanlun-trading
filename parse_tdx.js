const fs = require('fs');
const path = require('path');

const BASE_PATH = 'D:/大侠神器2.0/直接使用_大侠神器2.0.1.251231(ODM250901)/直接使用_大侠神器2.0.10B1206(260930)/new_tdx(V770)/vipdoc';

function readInt32LE(buffer, offset) {
    return buffer.readInt32LE(offset);
}

function readFloat(buffer, offset) {
    return buffer.readFloatLE(offset);
}

function parseDayFile(filepath) {
    const buffer = fs.readFileSync(filepath);
    const records = [];
    const recordSize = 32;

    for (let i = 0; i + recordSize <= buffer.length; i += recordSize) {
        const date = readInt32LE(buffer, i);
        const open = readInt32LE(buffer, i + 4) / 100;
        const high = readInt32LE(buffer, i + 8) / 100;
        const low = readInt32LE(buffer, i + 12) / 100;
        const close = readInt32LE(buffer, i + 16) / 100;
        const amount = readFloat(buffer, i + 20);
        const volume = readInt32LE(buffer, i + 24);
        // const reserved = readInt32LE(buffer, i + 28);

        records.push({
            date: date.toString(),
            open, high, low, close,
            amount, volume
        });
    }

    return records;
}

function parseMinFile(filepath) {
    const buffer = fs.readFileSync(filepath);
    const records = [];
    const recordSize = 32;

    for (let i = 0; i + recordSize <= buffer.length; i += recordSize) {
        const datetime = readInt32LE(buffer, i);
        const open = readInt32LE(buffer, i + 4) / 100;
        const high = readInt32LE(buffer, i + 8) / 100;
        const low = readInt32LE(buffer, i + 12) / 100;
        const close = readInt32LE(buffer, i + 16) / 100;
        const amount = readFloat(buffer, i + 20);
        const volume = readInt32LE(buffer, i + 24);

        records.push({
            datetime: datetime.toString(),
            open, high, low, close,
            amount, volume
        });
    }

    return records;
}

function toCSV(data, filepath) {
    if (data.length === 0) return;

    const headers = Object.keys(data[0]).join(',');
    const rows = data.map(r => Object.values(r).join(','));
    const csv = [headers, ...rows].join('\n');

    fs.writeFileSync(filepath, csv, 'utf8');
    console.log(`已保存: ${filepath} (${data.length}条记录)`);
}

// 批量导出所有日线数据
function exportAllDayFiles(outputDir = 'D:/新建文件夹/claude/tdx_data') {
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const markets = ['sh', 'sz', 'bj', 'ds', 'ot', 'cw'];
    let successCount = 0;
    let failCount = 0;

    markets.forEach(market => {
        const ldayPath = path.join(BASE_PATH, market, 'lday');
        if (!fs.existsSync(ldayPath)) return;

        const files = fs.readdirSync(ldayPath).filter(f => f.endsWith('.day'));
        console.log(`\n处理 ${market}/lday: ${files.length} 个文件`);

        files.forEach(file => {
            try {
                const inputPath = path.join(ldayPath, file);
                const data = parseDayFile(inputPath);
                if (data.length > 0) {
                    const outputFile = path.join(outputDir, `${file}.csv`);
                    toCSV(data, outputFile);
                    successCount++;
                }
            } catch (e) {
                console.error(`  失败: ${file} - ${e.message}`);
                failCount++;
            }
        });
    });

    console.log(`\n导出完成: 成功 ${successCount}, 失败 ${failCount}`);
}

function main() {
    console.log('=== 通达信数据解析器 (Node.js版) ===\n');

    // 检查命令行参数
    const args = process.argv.slice(1);
    const exportAll = args.includes('--all') || args.includes('-a');

    if (exportAll) {
        console.log('批量导出模式...\n');
        exportAllDayFiles();
        return;
    }

    // 解析上证指数
    const sh000001 = path.join(BASE_PATH, 'sh', 'lday', 'sh000001.day');
    if (fs.existsSync(sh000001)) {
        const data = parseDayFile(sh000001);
        console.log(`\n上证指数 (sh000001): ${data.length} 条记录`);
        console.log('最新5条:');
        data.slice(-5).forEach(r => {
            console.log(`  ${r.date} O:${r.open} H:${r.high} L:${r.low} C:${r.close} V:${r.volume}`);
        });
        toCSV(data, 'D:/新建文件夹/claude/sh000001.csv');
    }

    // 解析平安银行
    const sz000001 = path.join(BASE_PATH, 'sz', 'lday', 'sz000001.day');
    if (fs.existsSync(sz000001)) {
        const data = parseDayFile(sz000001);
        console.log(`\n平安银行 (sz000001): ${data.length} 条记录`);
        console.log('最新5条:');
        data.slice(-5).forEach(r => {
            console.log(`  ${r.date} O:${r.open} H:${r.high} L:${r.low} C:${r.close} V:${r.volume}`);
        });
        toCSV(data, 'D:/新建文件夹/claude/sz000001.csv');
    }

    // 扫描所有day文件
    console.log('\n=== 扫描所有日线文件 ===');
    const markets = ['sh', 'sz', 'bj', 'ds', 'ot', 'cw'];
    let totalFiles = 0;

    markets.forEach(market => {
        const ldayPath = path.join(BASE_PATH, market, 'lday');
        if (fs.existsSync(ldayPath)) {
            const files = fs.readdirSync(ldayPath).filter(f => f.endsWith('.day'));
            if (files.length > 0) {
                console.log(`${market}/lday: ${files.length} 个文件`);
                totalFiles += files.length;
            }
        }
    });

    console.log(`\n总计: ${totalFiles} 个日线文件`);
    console.log('\n使用方法:');
    console.log('  单个测试: node parse_tdx.js');
    console.log('  批量导出: node parse_tdx.js --all');
    console.log('\n分钟线数据: minline 目录为空，可能需要先下载分钟线数据');
    console.log('完成!');
}

main();
