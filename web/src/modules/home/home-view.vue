<script lang="ts" setup>
import { ref, watchEffect } from "vue";
import { watchDebounced } from "@vueuse/core";

const models = [
  "all-mpnet-base-v2",
  "all-MiniLM-L12-v2",
  "all-MiniLM-L6-v2",
  "paraphrase-multilingual-MiniLM-L12-v2",
  "paraphrase-MiniLM-L3-v2",
];

const query = ref("");
const selectedModel = ref("all-MiniLM-L12-v2");

const songs = ref({
  list: [],
  loading: false,
});

watchDebounced(
  [query, selectedModel],
  async () => {
    if (query.value === "") {
      songs.value = { list: [], loading: false };
    } else {
      songs.value = { list: [], loading: true };

      const model = selectedModel.value;
      const query_ = encodeURIComponent(query.value);

      const response = await fetch(`http://localhost:8000/query/${model}/5?query=${query_}`);
      const body = await response.json();
      songs.value = { list: body.map((item) => item.metadata.track_id), loading: false };
    }
  },
  { debounce: 500 }
);
</script>

<template>
  <main class="container">
    <h1>CrazzyFrogger</h1>

    <fieldset role="group">
      <input type="search" placeholder="Procure algo..." v-model="query" />
      <select aria-label="Modelo" v-model="selectedModel">
        <option v-for="model of models" :key="model">{{ model }}</option>
      </select>
    </fieldset>

    <span v-if="songs.loading" aria-busy="true">Searching songs...</span>

    <ul>
      <li v-for="song of songs.list" :key="song">
        <iframe
          style="border-radius: 12px"
          :src="`https://open.spotify.com/embed/track/${song}`"
          width="100%"
          height="80"
          frameBorder="0"
          allowfullscreen=""
          allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
          loading="lazy"
        ></iframe>
      </li>
    </ul>
  </main>
</template>

<style scoped lang="scss">
main {
  height: 100vh;
  padding: 2rem 1rem;
}

h1 {
  text-align: center;
  margin: 1em 0 1.5em;
}

input {
  margin-bottom: 2em;
}

ul {
  margin: 0;
  padding: 0;

  li {
    list-style-type: none;
    margin-bottom: 0.5em;
  }
}
</style>
