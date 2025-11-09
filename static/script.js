// UI interactions: theme, tabs, upload (fetch to /upload), prediction
const bodyEl = document.body;
document.getElementById("terang").addEventListener("click", () => bodyEl.classList.remove("dark"));
document.getElementById("gelap").addEventListener("click", () => bodyEl.classList.add("dark"));

// Tabs
const tabs = document.querySelectorAll(".tab");
const contents = document.querySelectorAll(".tab-content");
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    contents.forEach(c => c.classList.remove("active"));
    tab.classList.add("active");
    document.querySelector(tab.dataset.target).classList.add("active");
  });
});

// Fungsi untuk menampilkan detailed metrics
function displayDetailedMetrics(metrics) {
    console.log("Display detailed metrics:", metrics);
    
    // Decision Tree Detailed Report
    const dtBody = document.getElementById('dt_report_body');
    const dtFooter = document.getElementById('dt_report_footer');
    
    // Naive Bayes Detailed Report
    const nbBody = document.getElementById('nb_report_body');
    const nbFooter = document.getElementById('nb_report_footer');
    
    // Clear existing content
    if (dtBody) dtBody.innerHTML = '';
    if (nbBody) nbBody.innerHTML = '';
    if (dtFooter) dtFooter.innerHTML = '';
    if (nbFooter) nbFooter.innerHTML = '';
    
    // Populate Decision Tree table
    if (metrics.decision_tree && metrics.decision_tree.classes) {
        console.log("DT Classes:", metrics.decision_tree.classes);
        
        metrics.decision_tree.classes.forEach(cls => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${cls.name}</td>
                <td>${(cls.precision || 0).toFixed(2)}</td>
                <td>${(cls.recall || 0).toFixed(2)}</td>
                <td>${(cls.f1_score || 0).toFixed(2)}</td>
                <td>${cls.support || 0}</td>
            `;
            if (dtBody) dtBody.appendChild(row);
        });
        
        // Footer untuk Decision Tree
        if (metrics.decision_tree.macro_avg && metrics.decision_tree.weighted_avg && dtFooter) {
            dtFooter.innerHTML = `
                <tr>
                    <td><strong>Accuracy</strong></td>
                    <td colspan="3"></td>
                    <td><strong>${(metrics.decision_tree.accuracy || 0).toFixed(2)}</strong></td>
                </tr>
                <tr>
                    <td><strong>Macro Avg</strong></td>
                    <td>${(metrics.decision_tree.macro_avg.precision || 0).toFixed(2)}</td>
                    <td>${(metrics.decision_tree.macro_avg.recall || 0).toFixed(2)}</td>
                    <td>${(metrics.decision_tree.macro_avg.f1_score || 0).toFixed(2)}</td>
                    <td>${metrics.decision_tree.macro_avg.support || 0}</td>
                </tr>
                <tr>
                    <td><strong>Weighted Avg</strong></td>
                    <td>${(metrics.decision_tree.weighted_avg.precision || 0).toFixed(2)}</td>
                    <td>${(metrics.decision_tree.weighted_avg.recall || 0).toFixed(2)}</td>
                    <td>${(metrics.decision_tree.weighted_avg.f1_score || 0).toFixed(2)}</td>
                    <td>${metrics.decision_tree.weighted_avg.support || 0}</td>
                </tr>
            `;
        }
    } else {
        if (dtBody) dtBody.innerHTML = '<tr><td colspan="5" class="text-center">Tidak ada data</td></tr>';
    }
    
    // Populate Naive Bayes table
    if (metrics.naive_bayes && metrics.naive_bayes.classes) {
        console.log("NB Classes:", metrics.naive_bayes.classes);
        
        metrics.naive_bayes.classes.forEach(cls => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${cls.name}</td>
                <td>${(cls.precision || 0).toFixed(2)}</td>
                <td>${(cls.recall || 0).toFixed(2)}</td>
                <td>${(cls.f1_score || 0).toFixed(2)}</td>
                <td>${cls.support || 0}</td>
            `;
            if (nbBody) nbBody.appendChild(row);
        });
        
        // Footer untuk Naive Bayes
        if (metrics.naive_bayes.macro_avg && metrics.naive_bayes.weighted_avg && nbFooter) {
            nbFooter.innerHTML = `
                <tr>
                    <td><strong>Accuracy</strong></td>
                    <td colspan="3"></td>
                    <td><strong>${(metrics.naive_bayes.accuracy || 0).toFixed(2)}</strong></td>
                </tr>
                <tr>
                    <td><strong>Macro Avg</strong></td>
                    <td>${(metrics.naive_bayes.macro_avg.precision || 0).toFixed(2)}</td>
                    <td>${(metrics.naive_bayes.macro_avg.recall || 0).toFixed(2)}</td>
                    <td>${(metrics.naive_bayes.macro_avg.f1_score || 0).toFixed(2)}</td>
                    <td>${metrics.naive_bayes.macro_avg.support || 0}</td>
                </tr>
                <tr>
                    <td><strong>Weighted Avg</strong></td>
                    <td>${(metrics.naive_bayes.weighted_avg.precision || 0).toFixed(2)}</td>
                    <td>${(metrics.naive_bayes.weighted_avg.recall || 0).toFixed(2)}</td>
                    <td>${(metrics.naive_bayes.weighted_avg.f1_score || 0).toFixed(2)}</td>
                    <td>${metrics.naive_bayes.weighted_avg.support || 0}</td>
                </tr>
            `;
        }
    } else {
        if (nbBody) nbBody.innerHTML = '<tr><td colspan="5" class="text-center">Tidak ada data</td></tr>';
    }
}

// Fungsi untuk menampilkan analisis model
function displayModelAnalysis(metrics, detailedMetrics, targetColumn) {
    const analysisContainer = document.getElementById('modelAnalysis');
    if (!analysisContainer) {
        console.log("Element modelAnalysis tidak ditemukan");
        return;
    }

    let analysisHTML = '';
    
    // Analisis Performa Model
    analysisHTML += `
        <div class="analysis-section">
            <h6>🎯 Performa Model</h6>
            <div class="analysis-content">
    `;

    // Bandingkan akurasi
    const accDT = metrics.decision_tree.accuracy;
    const accNB = metrics.naive_bayes.accuracy;
    const f1DT = metrics.decision_tree.f1;
    const f1NB = metrics.naive_bayes.f1;

    // Box perbandingan
    analysisHTML += `
        <div class="model-comparison text-center">
            <h6>Perbandingan Akurasi</h6>
            <div class="row">
                <div class="col-6">
                    <h4>${(accDT * 100).toFixed(1)}%</h4>
                    <small>Decision Tree</small>
                </div>
                <div class="col-6">
                    <h4>${(accNB * 100).toFixed(1)}%</h4>
                    <small>Naive Bayes</small>
                </div>
            </div>
        </div>
    `;

    if (accDT > accNB) {
        analysisHTML += `
            <p><strong>Decision Tree lebih akurat</strong> (${(accDT * 100).toFixed(2)}% vs ${(accNB * 100).toFixed(2)}%)</p>
        `;
    } else if (accNB > accDT) {
        analysisHTML += `
            <p><strong>Naive Bayes lebih akurat</strong> (${(accNB * 100).toFixed(2)}% vs ${(accDT * 100).toFixed(2)}%)</p>
        `;
    } else {
        analysisHTML += `<p><strong>Kedua model memiliki akurasi yang sama</strong></p>`;
    }

    // Analisis F1-Score
    if (f1DT > f1NB) {
        analysisHTML += `<p><strong>Decision Tree memiliki F1-Score lebih baik</strong> (${f1DT.toFixed(4)} vs ${f1NB.toFixed(4)})</p>`;
    } else if (f1NB > f1DT) {
        analysisHTML += `<p><strong>Naive Bayes memiliki F1-Score lebih baik</strong> (${f1NB.toFixed(4)} vs ${f1DT.toFixed(4)})</p>`;
    }

    analysisHTML += `</div></div>`;

    // Analisis Kelas (jika detailed metrics tersedia)
    if (detailedMetrics && detailedMetrics.decision_tree && detailedMetrics.naive_bayes) {
        analysisHTML += `
            <div class="analysis-section mt-3">
                <h6>🔍 Analisis per Kelas</h6>
                <div class="analysis-content">
        `;

        const dtClasses = detailedMetrics.decision_tree.classes;
        const nbClasses = detailedMetrics.naive_bayes.classes;

        if (dtClasses && nbClasses && dtClasses.length > 0) {
            dtClasses.forEach((dtClass, index) => {
                const nbClass = nbClasses[index];
                if (dtClass && nbClass) {
                    analysisHTML += `
                        <div class="class-analysis">
                            <strong>Kelas: ${dtClass.name}</strong>
                            <div class="ms-3">
                                <small>
                                    <strong>Decision Tree:</strong> Precision=${dtClass.precision.toFixed(2)}, Recall=${dtClass.recall.toFixed(2)}, F1=${dtClass.f1_score.toFixed(2)}<br>
                                    <strong>Naive Bayes:</strong> Precision=${nbClass.precision.toFixed(2)}, Recall=${nbClass.recall.toFixed(2)}, F1=${nbClass.f1_score.toFixed(2)}
                                </small>
                            </div>
                        </div>
                    `;
                }
            });
        }

        analysisHTML += `</div></div>`;
    }

    // Rekomendasi Model
    analysisHTML += `
        <div class="analysis-section mt-3">
            <h6>💡 Rekomendasi</h6>
            <div class="analysis-content">
    `;

    const accuracyDiff = Math.abs(accDT - accNB);
    
    if (accuracyDiff < 0.05) {
        analysisHTML += `
            <p>Kedua model memiliki performa yang cukup seimbang. Pertimbangkan:</p>
            <ul>
                <li><strong>Decision Tree</strong> jika perlu interpretasi yang mudah dan memahami pola decision</li>
                <li><strong>Naive Bayes</strong> jika mengutamakan kecepatan dan dataset memiliki independensi fitur</li>
            </ul>
        `;
    } else if (accDT > accNB) {
        analysisHTML += `
            <p><strong>Rekomendasi: Gunakan Decision Tree</strong></p>
            <ul>
                <li>Lebih akurat dalam memprediksi ${targetColumn}</li>
                <li>Mampu menangkap hubungan non-linear dalam data</li>
                <li>Memberikan insight tentang feature importance</li>
            </ul>
        `;
    } else {
        analysisHTML += `
            <p><strong>Rekomendasi: Gunakan Naive Bayes</strong></p>
            <ul>
                <li>Lebih akurat dalam memprediksi ${targetColumn}</li>
                <li>Lebih cepat dalam training dan prediction</li>
                <li>Robust terhadap noise dalam data</li>
            </ul>
        `;
    }

    // Catatan tentang perbedaan prediksi
    analysisHTML += `
        <div class="alert alert-warning mt-2">
            <small>
                <strong>Catatan:</strong> Perbedaan prediksi antara model adalah hal normal. 
                Decision Tree cenderung lebih kompleks sedangkan Naive Bayes membuat asumsi independensi.
                Pilih model dengan akurasi tertinggi untuk prediksi yang lebih reliable.
            </small>
        </div>
    `;

    analysisHTML += `</div></div>`;

    analysisContainer.innerHTML = analysisHTML;
}

function preparePredictionInputs(columns) {
  const cont = document.getElementById("inputsContainer");
  if (!cont) {
    console.error("Element inputsContainer tidak ditemukan");
    return;
  }
  
  cont.innerHTML = "";
  // create an input for each column (simple heuristic)
  columns.forEach(c => {
    const div = document.createElement("div");
    div.style.display = "flex";
    div.style.flexDirection = "column";
    div.style.minWidth = "180px";
    const label = document.createElement("label"); 
    label.textContent = c;
    const input = document.createElement("input");
    input.name = c;
    input.placeholder = c;
    input.type = "text";
    div.appendChild(label);
    div.appendChild(input);
    cont.appendChild(div);
  });
}

// Upload handler
document.getElementById("btnUpload").addEventListener("click", async () => {
  const f = document.getElementById("fileUpload").files[0];
  const status = document.getElementById("uploadStatus");
  if (!f) { 
    status.textContent = "Silakan pilih file CSV dulu."; 
    return; 
  }
  
  status.textContent = "Mengunggah dan memproses...";
  const form = new FormData();
  form.append("file", f);

  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    const data = await res.json();
    
    console.log("Full response:", data); // Debug
    
    if (!res.ok) {
      status.textContent = "Gagal: " + (data.error || "error");
      return;
    }
    
    status.textContent = data.message || "Berhasil memproses dataset";

    // show sample table HTML
    const tableContainer = document.getElementById("tableContainer");
    if (tableContainer && data.sample_table) {
      tableContainer.innerHTML = data.sample_table;
    }

    // show metrics
    const metricsArea = document.getElementById("metricsArea");
    if (metricsArea && data.metrics) {
      metricsArea.innerHTML = `
        <div class="card">
          <h3>🌳 Decision Tree</h3>
          <p>Akurasi: ${(data.metrics.decision_tree.accuracy * 100).toFixed(2)}%</p>
          <p>F1-Skor: ${data.metrics.decision_tree.f1}</p>
        </div>
        <div class="card">
          <h3>🤖 Naive Bayes</h3>
          <p>Akurasi: ${(data.metrics.naive_bayes.accuracy * 100).toFixed(2)}%</p>
          <p>F1-Skor: ${data.metrics.naive_bayes.f1}</p>
        </div>
      `;
    }

    // Parse dan tampilkan detailed classification reports
    if (data.detailed_metrics) {
      console.log("Detailed metrics ditemukan:", data.detailed_metrics);
      displayDetailedMetrics(data.detailed_metrics);
      // Tampilkan analisis model
      displayModelAnalysis(data.metrics, data.detailed_metrics, data.target_column || "target");
    } else {
      console.log("Detailed metrics tidak tersedia dalam response");
      // Set fallback content
      const dtBody = document.getElementById('dt_report_body');
      const nbBody = document.getElementById('nb_report_body');
      if (dtBody) dtBody.innerHTML = '<tr><td colspan="5" class="text-center">Data detailed metrics tidak tersedia</td></tr>';
      if (nbBody) nbBody.innerHTML = '<tr><td colspan="5" class="text-center">Data detailed metrics tidak tersedia</td></tr>';
      
      // Tampilkan analisis dasar saja
      if (data.metrics) {
        displayModelAnalysis(data.metrics, null, data.target_column || "target");
      }
    }

    // show images
    const imagesArea = document.getElementById("imagesArea");
    if (imagesArea && data.images) {
      imagesArea.innerHTML = "";
      if (data.images.dt_confusion) {
        const img = document.createElement("img"); 
        img.src = data.images.dt_confusion; 
        img.alt = "Confusion Matrix Decision Tree";
        imagesArea.appendChild(img);
      }
      if (data.images.nb_confusion) {
        const img2 = document.createElement("img"); 
        img2.src = data.images.nb_confusion; 
        img2.alt = "Confusion Matrix Naive Bayes";
        imagesArea.appendChild(img2);
      }
      if (data.images.tree_plot) {
        const img3 = document.createElement("img"); 
        img3.src = data.images.tree_plot; 
        img3.alt = "Decision Tree Structure";
        imagesArea.appendChild(img3);
      }
    }

    // prepare prediction form inputs based on columns
    if (data.columns) {
      preparePredictionInputs(data.columns);
    }

  } catch (err) {
    console.error("Upload error:", err);
    status.textContent = "Terjadi error saat upload: " + err.message;
  }
});

// Prediction submit
document.getElementById("prediksiForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const inputs = form.querySelectorAll("input");
  const payload = {};
  
  inputs.forEach(i => {
    // try convert numeric
    const v = i.value;
    if (v === "") return;
    const n = Number(v);
    payload[i.name] = isNaN(n) ? v : n;
  });

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    const hasilPrediksi = document.getElementById("hasilPrediksi");
    
    if (!res.ok) {
      if (hasilPrediksi) {
        hasilPrediksi.textContent = "Error: " + (data.error || "Gagal prediksi");
      }
      return;
    }
    
    if (hasilPrediksi) {
      hasilPrediksi.innerHTML = `
        <h3>🌟 Hasil Prediksi:</h3>
        <p><b>Pohon Keputusan:</b> ${data.prediction.decision_tree}</p>
        <p><b>Naive Bayes:</b> ${data.prediction.naive_bayes}</p>
        <div class="alert alert-info mt-2">
            <small>
                <strong>Tips:</strong> Gunakan model dengan akurasi tertinggi dari hasil evaluasi di atas.
                Jika prediksi berbeda, periksa confusion matrix untuk memahami pola kesalahan model.
            </small>
        </div>
      `;
      hasilPrediksi.style.display = "block";
    }
    
  } catch (err) {
    console.error("Prediction error:", err);
    const hasilPrediksi = document.getElementById("hasilPrediksi");
    if (hasilPrediksi) {
      hasilPrediksi.textContent = "Terjadi error saat prediksi: " + err.message;
    }
  }
  
});