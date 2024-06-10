<script lang="ts" setup>
import { ref, watchEffect } from "vue";
import { watchDebounced } from "@vueuse/core";

const rankProfiles = [
  "track_name_semantic",
  "lyrics_semantic",
  "track_name_bm25",
  "lyrics_bm25",
];

const query = ref("");
const selectedRankProfile = ref("lyrics_semantic");

const songs = ref({
  list: [],
  loading: false,
});

watchDebounced(
  [query, selectedRankProfile],
  async () => {
    if (query.value === "") {
      songs.value = { list: [], loading: false };
    } else {
      songs.value = { list: [], loading: true };

      const rankProfile = selectedRankProfile.value;
      const query_ = encodeURIComponent(query.value);

      const response = await fetch(`http://localhost:8000/query/${rankProfile}/5?query=${query_}`);
      const body = await response.json();
      songs.value = { list: body.map((item) => item.track_id), loading: false };
    }
  },
  { debounce: 500 }
);
</script>

<template>
  <main class="container">
    <h1>CrazyFrogger</h1>

    <fieldset role="group">
      <input type="search" placeholder="Procure algo..." v-model="query" />
      <select aria-label="Modelo" v-model="selectedRankProfile">
        <option v-for="rankProfile of rankProfiles" :key="rankProfile">{{ rankProfile }}</option>
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
